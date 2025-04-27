# YOLO11 Feature Extractor

A utility for extracting both object detection results and intermediate layer features from YOLO11 models.

## Features

- Load any pretrained YOLO11 model (yolo11n.pt, yolo11s.pt, etc.)
- Process images individually or in batches with progress tracking
- Extract object detection results (bounding boxes, classes, confidence scores)
- Extract features from multiple intermediate layers simultaneously
- Normalize features using various methods (L2, min-max, standardization)
- Visualize detection results
- Save outputs in multiple formats (NumPy arrays, JSON)
- Command-line interface with comprehensive configuration options

## Prerequisites

```
pip install ultralytics torch numpy opencv-python pillow matplotlib tqdm
```

## Usage

### Basic Usage

```bash
python yolo11_feature_extractor.py --img-dir /path/to/images --output-dir yolo11_output
```

### With Feature Extraction

First, list available layers:

```bash
python yolo11_feature_extractor.py --list-layers
```

Then extract features from a specific layer:

```bash
python yolo11_feature_extractor.py --img-dir /path/to/images --output-dir yolo11_output --layer "10.conv"
```

Extract features from multiple layers:

```bash
python yolo11_feature_extractor.py --img-dir /path/to/images --layer "10.conv,11.conv,12.conv" --normalize l2
```

### All Options

```bash
python yolo11_feature_extractor.py --help
```

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