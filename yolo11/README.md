# YOLO11 - Vision LSF

This repository contains utilities for working with YOLO11 models, particularly focused on evaluation, feature extraction, and dataset validation.

## Contents

- `yolo_evaluator.py` - Evaluates YOLO model predictions against COCO ground truth annotations
- `yolo_eval_config.yaml` - Configuration file for the YOLO evaluator
- `dataset_validator.py` - Validates datasets for YOLO training and evaluation
- `yolo11_feature_extractor.py` - Extracts features and detections from YOLO11 models
- `yolo11_cache.py` - Caches model outputs to avoid repeated processing
- `yolo11_cache_demo.py` - Demonstrates caching functionality with the evaluator
- `test_yolo11.py` - Tests for YOLO11 functionality
- `YOLO_EVAL_README.md` - Detailed documentation for the YOLO evaluator

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/your-username/Vision_LSF.git
cd Vision_LSF/yolo11
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the YOLO evaluator:
```bash
python yolo_evaluator.py --config yolo_eval_config.yaml
```

## Feature Extraction

The `yolo11_feature_extractor.py` script allows you to extract features from any layer of a YOLO11 model:

```bash
python yolo11_feature_extractor.py --model yolo11n.pt --img-dir /path/to/images --output-dir /path/to/output --layer model.23
```

This will extract features from the specified layer for all images in the input directory.

## Model Output Caching

The `yolo11_cache.py` module provides functionality for caching model predictions to avoid repeated processing of the same images. This is particularly useful when you need to run the model multiple times on the same dataset.

### Using Caching with the Evaluator

The evaluator supports caching to speed up repeated evaluation runs:

```bash
# First run (creates cache)
python yolo_evaluator.py --config yolo_eval_config.yaml --enable-cache

# Second run (uses cache)
python yolo_evaluator.py --config yolo_eval_config.yaml --enable-cache
```

The caching functionality automatically:
1. Checks for cached predictions for each image
2. Uses cached predictions when available
3. Runs the model only for images not in the cache
4. Saves new predictions to the cache

### Caching Logits for Conformal Prediction

For conformal prediction workflows, you can also cache the raw model outputs (logits):

```bash
python yolo_evaluator.py --config yolo_eval_config.yaml --enable-cache --cache-logits
```

This will store:
1. The final model predictions (bounding boxes, class IDs, confidence scores)
2. Raw model outputs before post-processing (confidence logits, class logits, etc.)

These raw outputs can then be used for conformal prediction or other uncertainty quantification methods.

### Cache Configuration

You can configure the cache directory and logit caching in the `yolo_eval_config.yaml` file:

```yaml
# Cache configuration
cache:
  dir: 'yolo11_cache'   # Directory to store cache files
  cache_logits: false    # Whether to cache raw model outputs for conformal prediction
  logits_to_extract: ['probs', 'conf', 'cls']  # Types of logits to extract and cache
```

### Cache Demonstration

To see a demonstration of the caching functionality and measure the speedup:

```bash
python yolo11_cache_demo.py --config yolo_eval_config.yaml --num-images 10
```

This script:
1. Runs the evaluator without caching
2. Runs the evaluator with caching
3. Measures and reports the speedup

### Using the Cache API Directly

You can also use the caching API directly in your own code:

```python
from yolo11_cache import YOLO11Cache

# Initialize cache with logit caching enabled
cache = YOLO11Cache(
    cache_dir="yolo11_cache",
    model_name="yolo11n.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
    cache_logits=True  # Enable logit caching
)

# Load cached predictions
cache.load_cache("val_dataset")

# Check if prediction exists for an image
if cache.has_prediction("image.jpg"):
    # Get cached prediction with logits
    prediction, logits = cache.get_prediction("image.jpg", include_logits=True)
    
    # Use prediction and logits...
    # Logits can be used for conformal prediction or uncertainty quantification
else:
    # Run model and add to cache
    # ...
    cache.save_prediction("image.jpg", prediction, logits=model_logits)

# Save updated cache
cache.save_cache("val_dataset")
```

## Dataset Validation

The `dataset_validator.py` script helps validate datasets for YOLO training and evaluation:

```bash
python dataset_validator.py --dataset-dir /path/to/dataset --annotations /path/to/annotations.json
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
usage: yolo11_feature_extractor.py [-h] [--model MODEL] --img-dir IMG_DIR
                                   [--output-dir OUTPUT_DIR] [--layer LAYER]
                                   [--list-layers] [--batch-size BATCH_SIZE]
                                   [--conf CONF] [--iou IOU] [--device DEVICE]
                                   [--max-images MAX_IMAGES]
                                   [--output-format {numpy,json,both}]
                                   [--normalize {l2,min_max,standardize}]
                                   [--no-viz]

YOLO11 Feature Extractor

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         YOLO11 model to use
  --img-dir IMG_DIR     Directory containing images
  --output-dir OUTPUT_DIR
                        Output directory
  --layer LAYER         Layer to extract features from (can be comma-separated for multiple layers)
  --list-layers         List available layers and exit
  --batch-size BATCH_SIZE
                        Batch size for processing
  --conf CONF           Confidence threshold
  --iou IOU             IOU threshold
  --device DEVICE       Device to run on (cuda, cpu)
  --max-images MAX_IMAGES
                        Maximum number of images to process
  --output-format {numpy,json,both}
                        Format for saving detections (numpy, json, or both)
  --normalize {l2,min_max,standardize}
                        Normalize features using specified method
  --no-viz              Disable visualization saving
```

## Output Structure

The script creates the following directory structure in the output directory:

```
output_dir/
  ├── detections/          # Contains object detection results (numpy arrays and/or JSON files)
  ├── features/            # Contains extracted features (PyTorch tensors)
  ├── visualizations/      # Contains visualization images with bounding boxes
  └── examples.png         # Summary visualization of selected examples
```

## Example Code

```python
from yolo11_feature_extractor import YOLO11FeatureExtractor

# Initialize feature extractor
extractor = YOLO11FeatureExtractor(
    model_name="yolo11n.pt",
    device="cuda",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# List available layers
layers = extractor.get_available_layers()
print("Available layers:", layers)

# Process images with multiple layer extraction and feature normalization
results = extractor.process_images(
    img_dir="path/to/images",
    output_dir="output",
    layer_names=["10.conv", "11.conv"],  # Extract from multiple layers
    batch_size=4,
    output_format="both",  # Save in both NumPy and JSON formats
    normalize_features="l2"  # Apply L2 normalization to features
)

# Print results
for img_name, result in results.items():
    print(f"Image: {img_name}")
    print(f"  Detections: {len(result['detections'])}")
    for layer, shape in result['feature_shapes'].items():
        print(f"  Features from {layer}: shape={shape}")
``` 