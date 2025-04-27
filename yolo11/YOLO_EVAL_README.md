# YOLO11 Evaluation Tool

A comprehensive tool for evaluating YOLO11 models against COCO ground truth annotations. The tool calculates precision, recall, F1 score, and other metrics, and generates visualizations of detection results.

## Features

- Compare YOLO11 predictions with COCO ground truth annotations
- Calculate overall and per-class metrics (precision, recall, F1 score)
- Generate visualizations showing true positives, false positives, and false negatives
- Configure all aspects of evaluation through a YAML configuration file
- Support for evaluating on specific images or entire datasets
- Export metrics to CSV for further analysis

## Installation

### Prerequisites

```bash
pip install ultralytics numpy opencv-python matplotlib pandas pyyaml tqdm
```

### Files

The evaluation tool consists of three key files:
- `yolo_evaluator.py` - The main evaluation script
- `yolo_eval_config.yaml` - Configuration file for the evaluation
- `YOLO_EVAL_README.md` - This README file

## Usage

### Basic Usage

```bash
python yolo_evaluator.py --config yolo_eval_config.yaml
```

### Configuration

The `yolo_eval_config.yaml` file contains all settings for the evaluation. Here's a brief explanation of the key sections:

1. **Paths**: Define paths to COCO dataset and output directory
2. **Model**: Configure the YOLO model to evaluate
3. **Evaluation parameters**: Set IoU threshold, specific images to evaluate, etc.
4. **Visualization settings**: Configure visualization options
5. **Metrics**: Define which metrics to calculate

The full configuration file is well-documented with comments explaining each option.

### Evaluating Specific Images

To evaluate specific images:

1. Set `eval_full_dataset: false` in the config file
2. Add image filenames to the `specific_images` list:
   ```yaml
   specific_images: 
     - "COCO_train2014_000000000064.jpg"
     - "COCO_train2014_000000000641.jpg"
   ```

### Evaluating Full Dataset

To evaluate the entire dataset:

1. Set `eval_full_dataset: true` in the config file
2. Optionally set `max_images` to limit the number of images to evaluate:
   ```yaml
   max_images: 1000  # Evaluate first 1000 images only
   ```

## Output

The tool creates several output files in the specified output directory:

1. **Visualizations**: Annotated images showing ground truth, true positives, false positives, and false negatives
2. **Metrics**: CSV files with overall and per-class metrics
3. **Console output**: Summary of evaluation results

## Metrics

The tool calculates the following metrics:

- **Precision**: Fraction of detected objects that are correct
- **Recall**: Fraction of ground truth objects that are correctly detected
- **F1 Score**: Harmonic mean of precision and recall
- **Average IoU**: Average Intersection over Union for true positives
- **Per-class metrics**: Precision, recall, and F1 score for each class

## Example Visualization

The visualization shows:
- Ground truth boxes (green)
- True positive detections (blue)
- False positive detections (red)
- False negative ground truths (yellow dashed)

Each box is labeled with its class, and true positives include confidence score and IoU value.

## Customization

The tool is designed to be easily customizable through the YAML configuration file. You can adjust:

- IoU threshold for matching
- Confidence threshold for predictions
- Visualization settings (colors, line thickness, etc.)
- Which metrics to calculate and save

## Troubleshooting

If you encounter issues:

1. **Image not found**: Ensure paths in the config file are correct
2. **CUDA out of memory**: Reduce batch size or use CPU device
3. **Empty annotations**: Check that annotation files are correctly loaded

## License

This tool is provided under the MIT License.

## Citation

If you use this tool in your research, please cite the Ultralytics YOLO project:

```
@misc{ultralyticsyolo11,
  title={{Ultralytics YOLOv11}},
  author={Jocher, Glenn and others},
  year={2023},
  organization={Ultralytics},
  url={https://github.com/ultralytics/ultralytics}
}
``` 