#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO Model Evaluator

This script evaluates YOLO model predictions against COCO ground truth annotations.
It calculates precision, recall, F1 score, and mAP metrics at various IoU thresholds.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import torch  # Add torch import for the type annotations
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from ultralytics import YOLO

# Import caching functionality
from yolo11_cache import YOLO11Cache

class COCOEvaluator:
    """Class for evaluating YOLO model predictions against COCO ground truth."""
    
    def __init__(self, config_path: str, enable_cache: bool = False, cache_logits: bool = False):
        """
        Initialize the YOLO evaluator.
        
        Args:
            config_path: Path to the YAML configuration file
            enable_cache: Whether to enable caching of model predictions
            cache_logits: Whether to cache raw model outputs (logits) for conformal prediction
        """
        # Load configuration file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.train_images_dir = self.config['paths']['coco_dataset']['train_images']
        self.val_images_dir = self.config['paths']['coco_dataset']['val_images']
        self.train_annotations_file = self.config['paths']['coco_dataset']['train_annotations']
        self.val_annotations_file = self.config['paths']['coco_dataset']['val_annotations']
        self.output_dir = self.config['paths']['output_dir']
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.vis_dir = os.path.join(self.output_dir, "visualizations")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Set evaluation parameters
        self.dataset_split = self.config['evaluation'].get('dataset_split', 'train')  # Default to train if not specified
        self.match_iou_threshold = self.config['evaluation']['match_iou_threshold']
        self.eval_full_dataset = self.config['evaluation']['eval_full_dataset']
        self.specific_images = self.config['evaluation']['specific_images']
        self.max_images = self.config['evaluation']['max_images']
        
        # Set up model
        self.model_name = self.config['model']['name']
        self.device = self.config['model']['device']
        self.conf_threshold = self.config['model']['conf_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        
        # Load the model
        self.model = None
        self.coco_gt_data = None
        self.categories = None
        self.category_mapping = None
        
        # Class mapping from YOLO to COCO (will be initialized later)
        self.yolo_to_coco_mapping = None
        
        # Caching setup
        self.enable_cache = enable_cache
        self.cache_logits = cache_logits
        self.cache = None
        if enable_cache:
            cache_dir = self.config.get('cache', {}).get('dir', 'yolo11_cache')
            self.cache = YOLO11Cache(
                cache_dir=cache_dir,
                model_name=self.model_name,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                cache_logits=cache_logits
            )
            print(f"Caching enabled. Cache directory: {cache_dir}")
            if cache_logits:
                print("Logit caching enabled for conformal prediction.")
    
    def load_model(self):
        """Load the YOLO model."""
        print(f"Loading model {self.model_name}...")
        try:
            # Check if model exists at a specific location first
            local_model_path = os.path.join('/ssd_4TB/divake/Vision_LSF/yolo11', self.model_name)
            if os.path.exists(local_model_path):
                print(f"Using existing model at {local_model_path}")
                self.model = YOLO(local_model_path)
            else:
                print(f"Model not found at {local_model_path}, using default path or downloading")
                self.model = YOLO(self.model_name)
            print(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_coco_annotations(self, is_train: bool = True) -> Tuple[Dict, Dict]:
        """
        Load COCO annotations.
        
        Args:
            is_train: Whether to load train or validation annotations
            
        Returns:
            Tuple of (annotations data, category mapping)
        """
        annotation_file = self.train_annotations_file if is_train else self.val_annotations_file
        print(f"Loading COCO annotations from {annotation_file}...")
        
        try:
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
                
            # Create category ID to name mapping
            categories = {}
            for cat in coco_data['categories']:
                categories[cat['id']] = cat['name']
            
            print(f"Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
            print(f"Categories: {len(categories)}")
            
            # Create mapping between YOLO class IDs and COCO category IDs
            # This is needed because YOLO and COCO might use different class IDs for the same class names
            if self.model:
                self.create_class_mapping(categories)
            
            return coco_data, categories
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            raise
    
    def create_class_mapping(self, coco_categories: Dict):
        """
        Create mapping between YOLO class IDs and COCO category IDs based on matching names.
        
        Args:
            coco_categories: Dictionary mapping COCO category IDs to category names
        """
        print("Creating class mapping between YOLO and COCO...")
        self.yolo_to_coco_mapping = {}
        
        # Create dictionaries for name-based lookup
        yolo_names_lower = {name.lower(): id for id, name in self.model.names.items()}
        coco_names_lower = {name.lower(): id for id, name in coco_categories.items()}
        
        # Find matching class names
        matched_count = 0
        for coco_id, coco_name in coco_categories.items():
            coco_name_lower = coco_name.lower()
            if coco_name_lower in yolo_names_lower:
                yolo_id = yolo_names_lower[coco_name_lower]
                self.yolo_to_coco_mapping[yolo_id] = coco_id
                matched_count += 1
        
        print(f"Mapped {matched_count} out of {len(coco_categories)} COCO categories to YOLO classes")
        
        if matched_count == 0:
            print("⚠️ Warning: No matching categories found between YOLO and COCO!")
            print("This will result in zero metrics. Check class names for exact matches.")
        elif matched_count < len(coco_categories) * 0.5:
            print("⚠️ Warning: Less than 50% of categories matched. Metrics may be affected.")
    
    def get_image_ground_truth(self, image_file: str, coco_data: Dict) -> List[Dict]:
        """
        Get ground truth annotations for a specific image.
        
        Args:
            image_file: Image filename (e.g., 'COCO_train2014_000000000064.jpg')
            coco_data: Loaded COCO annotations
            
        Returns:
            List of ground truth objects in the format:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'category_id': id,
                    'category_name': name
                },
                ...
            ]
        """
        # Find image ID
        image_id = None
        image_width = None
        image_height = None
        
        for img in coco_data['images']:
            if img['file_name'] == image_file:
                image_id = img['id']
                image_width = img['width']
                image_height = img['height']
                break
        
        if image_id is None:
            print(f"Warning: Image {image_file} not found in annotations")
            return []
        
        # Get annotations for this image
        gt_objects = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                # Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]
                x, y, w, h = ann['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                category_id = ann['category_id']
                category_name = self.categories.get(category_id, 'unknown')
                
                gt_objects.append({
                    'bbox': [x1, y1, x2, y2],
                    'category_id': category_id,
                    'category_name': category_name
                })
        
        return gt_objects, (image_width, image_height)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First box coordinates [x1, y1, x2, y2]
            box2: Second box coordinates [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def match_detections_to_ground_truth(
        self, 
        detections: List[Dict], 
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Match detected objects to ground truth objects.
        
        Args:
            detections: List of detected objects
            ground_truth: List of ground truth objects
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Tuple of (true positives, false positives, false negatives)
        """
        true_positives = []  # Correct detections
        false_positives = []  # Wrong detections
        
        # Make a copy of ground truth to track matches
        unmatched_gt = ground_truth.copy()
        
        # For each detection, find the best matching ground truth
        for det in detections:
            det_bbox = det['bbox']
            det_class = det['class_id']
            
            # Map YOLO class ID to COCO category ID if mapping exists
            if self.yolo_to_coco_mapping and det_class in self.yolo_to_coco_mapping:
                det_class_mapped = self.yolo_to_coco_mapping[det_class]
            else:
                det_class_mapped = det_class
            
            best_match = None
            best_iou = 0
            best_idx = -1
            
            # Find ground truth with highest IoU
            for i, gt in enumerate(unmatched_gt):
                gt_bbox = gt['bbox']
                gt_class = gt['category_id']
                
                # Calculate IoU
                iou = self.calculate_iou(det_bbox, gt_bbox)
                
                # Check if this is the best match so far
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt
                    best_idx = i
            
            # Check if match is good enough (above threshold) and classes match
            if best_match and best_iou >= iou_threshold:
                # For a true positive, the IoU must be above threshold AND classes must match
                # Note: We need to check if the YOLO class ID maps to the COCO category ID
                if det_class_mapped == best_match['category_id']:
                    # This is a true positive
                    true_positives.append({
                        'detection': det,
                        'ground_truth': best_match,
                        'iou': best_iou
                    })
                    # Remove matched ground truth from unmatched list
                    unmatched_gt.pop(best_idx)
                else:
                    # Class mismatch is a false positive
                    false_positives.append({
                        'detection': det,
                        'closest_gt': best_match,
                        'iou': best_iou
                    })
            else:
                # No good match found - false positive
                false_positives.append({
                    'detection': det,
                    'closest_gt': best_match,
                    'iou': best_iou
                })
        
        # Any remaining unmatched ground truth objects are false negatives
        false_negatives = [{'ground_truth': gt} for gt in unmatched_gt]
        
        return true_positives, false_positives, false_negatives
    
    def calculate_metrics(
        self, 
        true_positives: List[Dict], 
        false_positives: List[Dict], 
        false_negatives: List[Dict]
    ) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            true_positives: List of true positive matches
            false_positives: List of false positive detections
            false_negatives: List of false negative ground truths
            
        Returns:
            Dictionary of metrics
        """
        # Count TP, FP, FN
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        
        # Calculate precision, recall, F1
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate average IoU for true positives
        avg_iou = np.mean([tp['iou'] for tp in true_positives]) if true_positives else 0
        
        # Calculate per-class metrics
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Count TP and FP by class
        for tp in true_positives:
            class_name = tp['detection']['class_name']
            class_metrics[class_name]['tp'] += 1
        
        for fp in false_positives:
            class_name = fp['detection']['class_name']
            class_metrics[class_name]['fp'] += 1
        
        # Count FN by class
        for fn in false_negatives:
            class_name = fn['ground_truth']['category_name']
            class_metrics[class_name]['fn'] += 1
        
        # Calculate per-class precision, recall, F1
        for class_name, counts in class_metrics.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            
            counts['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            counts['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            counts['f1'] = 2 * counts['precision'] * counts['recall'] / (counts['precision'] + counts['recall']) if (counts['precision'] + counts['recall']) > 0 else 0
        
        # Calculate mAP at different IoU thresholds
        map_metrics = self.calculate_map_at_iou_thresholds(true_positives, false_positives, false_negatives)
        
        return {
            'overall': {
                'true_positives': tp_count,
                'false_positives': fp_count,
                'false_negatives': fn_count,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_iou': avg_iou,
                'map_metrics': map_metrics
            },
            'per_class': dict(class_metrics)
        }
    
    def calculate_map_at_iou_thresholds(
        self,
        true_positives: List[Dict], 
        false_positives: List[Dict], 
        false_negatives: List[Dict],
        iou_thresholds=[0.5, 0.75, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ) -> Dict:
        """
        Calculate mAP at different IoU thresholds.
        
        Args:
            true_positives: List of true positive matches
            false_positives: List of false positive detections
            false_negatives: List of false negative ground truths
            iou_thresholds: List of IoU thresholds to calculate mAP for
            
        Returns:
            Dictionary of mAP metrics
        """
        # Initialize mAP metrics
        map_metrics = {
            'map50': 0.0,      # mAP at IoU=0.5
            'map75': 0.0,      # mAP at IoU=0.75
            'map50-95': 0.0    # mAP at IoU=0.5:0.95
        }
        
        # Calculate mAP at IoU=0.5
        if 0.5 in iou_thresholds:
            # For simplicity in this implementation, we'll use the standard precision, recall
            # True mAP calculation would involve calculating the area under the precision-recall curve
            # This is a simplified version to demonstrate the concept
            map_metrics['map50'] = self.calculate_average_precision(true_positives, false_positives, false_negatives, 0.5)
        
        # Calculate mAP at IoU=0.75
        if 0.75 in iou_thresholds:
            map_metrics['map75'] = self.calculate_average_precision(true_positives, false_positives, false_negatives, 0.75)
        
        # Calculate mAP at IoU=0.5:0.95
        # This is the average of mAP calculated at IoU thresholds from 0.5 to 0.95 with a step of 0.05
        if set([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]).issubset(set(iou_thresholds)):
            map_values = []
            for iou_threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                map_values.append(self.calculate_average_precision(true_positives, false_positives, false_negatives, iou_threshold))
            
            map_metrics['map50-95'] = np.mean(map_values) if map_values else 0.0
            
        return map_metrics
    
    def calculate_average_precision(
        self,
        true_positives: List[Dict], 
        false_positives: List[Dict], 
        false_negatives: List[Dict],
        iou_threshold: float
    ) -> float:
        """
        Calculate average precision at a specific IoU threshold.
        
        Args:
            true_positives: List of true positive matches
            false_positives: List of false positive detections
            false_negatives: List of false negative ground truths
            iou_threshold: IoU threshold to calculate AP for
            
        Returns:
            Average precision value
        """
        # Filter true positives based on IoU threshold
        tp_at_threshold = [tp for tp in true_positives if tp['iou'] >= iou_threshold]
        
        # Count TP, FP, FN at this threshold
        tp_count = len(tp_at_threshold)
        fp_count = len(false_positives) + (len(true_positives) - tp_count)  # FPs + TPs that became FPs
        fn_count = len(false_negatives)
        
        # Calculate precision, recall, and AP
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        
        # For simplicity, we use precision as a proxy for AP
        # A true AP calculation would calculate the area under the precision-recall curve
        # This is a simplified version for demonstration
        average_precision = precision
        
        return average_precision
    
    def visualize_results(
        self, 
        image_path: str,
        image_size: Tuple[int, int],
        ground_truth: List[Dict],
        true_positives: List[Dict],
        false_positives: List[Dict],
        false_negatives: List[Dict],
        output_path: str
    ) -> np.ndarray:
        """
        Create visualization of detection results.
        
        Args:
            image_path: Path to the input image
            image_size: (width, height) of the image
            ground_truth: List of ground truth objects
            true_positives: List of true positive matches
            false_positives: List of false positive detections
            false_negatives: List of false negative ground truths
            output_path: Path to save the visualization
            
        Returns:
            Annotated image as numpy array
        """
        # Load the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        except Exception as e:
            print(f"Error loading image for visualization: {e}")
            # Create a blank image with the specified size if original can't be loaded
            image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Get colors from config
        vis_config = self.config['visualization']
        gt_color = tuple(vis_config['colors']['ground_truth'])
        tp_color = tuple(vis_config['colors']['true_positive'])
        fp_color = tuple(vis_config['colors']['false_positive'])
        fn_color = tuple(vis_config['colors']['false_negative'])
        
        # Get line thickness
        box_thickness = vis_config['box_thickness']
        
        # Get font settings
        font_scale = vis_config['font']['scale']
        font_thickness = vis_config['font']['thickness']
        font_color = tuple(vis_config['font']['color'])
        
        # Draw ground truth boxes if enabled
        if vis_config['draw_ground_truth']:
            for gt in ground_truth:
                bbox = gt['bbox']
                label = gt['category_name']
                
                # Draw box
                cv2.rectangle(
                    image, 
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    gt_color, 
                    box_thickness
                )
                
                # Draw label
                cv2.putText(
                    image, 
                    f"GT: {label}", 
                    (int(bbox[0]), int(bbox[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    font_color, 
                    font_thickness
                )
        
        # Draw predicted boxes if enabled
        if vis_config['draw_predictions']:
            # Draw true positives
            for tp in true_positives:
                bbox = tp['detection']['bbox']
                label = tp['detection']['class_name']
                confidence = tp['detection']['confidence']
                iou = tp['iou']
                
                # Draw box
                cv2.rectangle(
                    image, 
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    tp_color, 
                    box_thickness
                )
                
                # Draw label
                cv2.putText(
                    image, 
                    f"{label} {confidence:.2f} IoU:{iou:.2f}", 
                    (int(bbox[0]), int(bbox[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    font_color, 
                    font_thickness
                )
            
            # Draw false positives
            for fp in false_positives:
                bbox = fp['detection']['bbox']
                label = fp['detection']['class_name']
                confidence = fp['detection']['confidence']
                
                # Draw box
                cv2.rectangle(
                    image, 
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    fp_color, 
                    box_thickness
                )
                
                # Draw label
                cv2.putText(
                    image, 
                    f"FP: {label} {confidence:.2f}", 
                    (int(bbox[0]), int(bbox[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    font_color, 
                    font_thickness
                )
            
            # Draw false negatives
            for fn in false_negatives:
                bbox = fn['ground_truth']['bbox']
                label = fn['ground_truth']['category_name']
                
                # Draw box with dashed lines
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Draw dashed lines for false negatives
                dash_length = 10
                for i in range(0, x2 - x1, dash_length * 2):
                    cv2.line(image, (x1 + i, y1), (min(x1 + i + dash_length, x2), y1), fn_color, box_thickness)
                    cv2.line(image, (x1 + i, y2), (min(x1 + i + dash_length, x2), y2), fn_color, box_thickness)
                
                for i in range(0, y2 - y1, dash_length * 2):
                    cv2.line(image, (x1, y1 + i), (x1, min(y1 + i + dash_length, y2)), fn_color, box_thickness)
                    cv2.line(image, (x2, y1 + i), (x2, min(y1 + i + dash_length, y2)), fn_color, box_thickness)
                
                # Draw label
                cv2.putText(
                    image, 
                    f"FN: {label}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    font_color, 
                    font_thickness
                )
        
        # Add summary at the bottom of the image
        metrics = self.calculate_metrics(true_positives, false_positives, false_negatives)
        summary_text = f"Precision: {metrics['overall']['precision']:.2f}, Recall: {metrics['overall']['recall']:.2f}, F1: {metrics['overall']['f1_score']:.2f}"
        cv2.putText(
            image,
            summary_text,
            (10, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness
        )
        
        # Save the image
        if vis_config['save']:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to {output_path}")
        
        # Show the image if requested
        if vis_config['show']:
            try:
                # Use matplotlib for display (more portable than cv2.imshow)
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"Detection Results - {os.path.basename(image_path)}")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display visualization: {e}")
        
        return image
    
    def save_metrics_to_csv(self, metrics: Dict, output_path: str):
        """Save metrics to CSV file."""
        # Overall metrics
        overall_df = pd.DataFrame([metrics['overall']])
        overall_path = os.path.join(output_path, "overall_metrics.csv")
        overall_df.to_csv(overall_path, index=False)
        
        # Per-class metrics
        class_data = []
        for class_name, class_metrics in metrics['per_class'].items():
            row = {'class': class_name}
            row.update(class_metrics)
            class_data.append(row)
        
        class_df = pd.DataFrame(class_data)
        class_path = os.path.join(output_path, "per_class_metrics.csv")
        class_df.to_csv(class_path, index=False)
        
        print(f"Metrics saved to {output_path}")
    
    def extract_logits(self, results) -> Dict[str, torch.Tensor]:
        """
        Extract raw logits from YOLO model predictions for conformal prediction.
        
        Args:
            results: YOLO model prediction results
            
        Returns:
            Dictionary containing raw logits
        """
        # Get the result for the image
        result = results[0]
        
        # Extract tensors before NMS was applied
        logits = {}
        
        try:
            # Extract the tensors we care about for conformal prediction
            # The exact structure depends on the YOLO version and model
            
            # If available, get pre-NMS outputs
            if hasattr(result, 'probs') and result.probs is not None:
                logits['probs'] = result.probs.detach().clone()
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Save the raw detection boxes
                if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                    logits['conf'] = result.boxes.conf.detach().clone()
                
                # Class probabilities before argmax was applied
                if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                    logits['cls'] = result.boxes.cls.detach().clone()
            
            # Try to get proto or other outputs if available
            if hasattr(result, 'proto') and result.proto is not None:
                logits['proto'] = result.proto.detach().clone()
            
            # Add any other tensors needed for conformal prediction
            # ...
            
        except Exception as e:
            print(f"Warning: Error extracting logits: {e}")
        
        return logits
    
    def evaluate_image(self, image_path: str, ground_truth: List[Dict], image_size: Tuple[int, int]) -> Dict:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to the image
            ground_truth: List of ground truth objects
            
        Returns:
            Evaluation metrics
        """
        # Check cache first if enabled
        cached_prediction = None
        cached_logits = None
        image_filename = os.path.basename(image_path)
        
        if self.enable_cache and self.cache:
            if self.cache.has_prediction(image_filename):
                if self.cache_logits:
                    cached_prediction, cached_logits = self.cache.get_prediction(image_filename, include_logits=True)
                    print(f"Using cached prediction and logits for {image_filename}")
                else:
                    cached_prediction = self.cache.get_prediction(image_filename)
                    print(f"Using cached prediction for {image_filename}")
        
        if cached_prediction is not None:
            # Use the cached prediction
            detections = []
            for box in cached_prediction.get('boxes', []):
                detections.append(box)
        else:
            # Run prediction with the model
            results = self.model(
                image_path, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                device=self.device, 
                verbose=False
            )
            
            # Extract logits for conformal prediction if enabled
            extracted_logits = None
            if self.cache_logits:
                extracted_logits = self.extract_logits(results)
            
            # Get the detection results
            result = results[0]
            boxes = result.boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
            
            # Convert to our format
            detections = []
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': self.model.names[cls_id]
                }
                detections.append(detection)
            
            # Cache the prediction if caching is enabled
            if self.enable_cache and self.cache:
                # Save ground truth along with predictions
                cached_data = {
                    'boxes': detections,
                    'ground_truth': ground_truth,
                    'image_size': image_size
                }
                self.cache.save_prediction(image_filename, cached_data, extracted_logits)
        
        # Match detections to ground truth
        true_positives, false_positives, false_negatives = self.match_detections_to_ground_truth(
            detections, 
            ground_truth, 
            iou_threshold=self.match_iou_threshold
        )
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_positives, false_positives, false_negatives)
        
        # Create visualization
        if self.config['visualization']['enabled']:
            vis_filename = os.path.basename(image_path).split('.')[0] + '_eval.jpg'
            vis_path = os.path.join(self.vis_dir, vis_filename)
            
            self.visualize_results(
                image_path=image_path,
                image_size=image_size,
                ground_truth=ground_truth,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                output_path=vis_path
            )
        
        return {
            'metrics': metrics,
            'matches': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        }
    
    def split_for_calibration(self, image_files, coco_data):
        """
        Split validation data into validation and calibration sets,
        ensuring class balance between the two sets.
        
        Args:
            image_files: List of (filename, path) tuples for validation images
            coco_data: COCO annotations data
            
        Returns:
            Tuple of (validation_files, calibration_files)
        """
        print("Splitting validation data into validation and calibration sets...")
        
        # Use a different approach to get a more balanced split
        # Sort all images by their class distributions
        images_with_classes = []
        
        # Process each image to get its class distribution
        for img_file, img_path in image_files:
            # Get ground truth for this image
            ground_truth, _ = self.get_image_ground_truth(img_file, coco_data)
            
            if not ground_truth:
                continue
                
            # Get class counts for this image
            class_ids = [gt['category_id'] for gt in ground_truth]
            
            # Add to our list with class information
            images_with_classes.append((img_file, img_path, class_ids))
        
        # Shuffle the images first to randomize
        import random
        random.shuffle(images_with_classes)
        
        # Sort images by number of classes first (to interleave simple and complex images)
        images_with_classes.sort(key=lambda x: len(x[2]))
        
        # Split into two groups ensuring even distribution
        val_files = []
        cal_files = []
        
        # Use alternating assignment to split evenly
        for i, (img_file, img_path, _) in enumerate(images_with_classes):
            if i % 2 == 0:
                val_files.append((img_file, img_path))
            else:
                cal_files.append((img_file, img_path))
        
        # Double-check we have close to 50-50 split
        total = len(val_files) + len(cal_files)
        val_pct = len(val_files) / total * 100 if total > 0 else 0
        cal_pct = len(cal_files) / total * 100 if total > 0 else 0
        
        print(f"Split completed: {len(val_files)} images ({val_pct:.1f}%) for validation, "
              f"{len(cal_files)} images ({cal_pct:.1f}%) for calibration")
        
        # If the split is very unbalanced (more than 60-40), warn the user
        if abs(len(val_files) - len(cal_files)) > 0.2 * total:
            print("WARNING: The validation/calibration split is not well balanced. "
                  "This may affect model evaluation and calibration.")
        
        return val_files, cal_files

    def run_evaluation(self):
        """Run the evaluation process."""
        # Load model
        self.load_model()
        
        # Determine which dataset to use based on configuration
        is_train = self.dataset_split.lower() == 'train'
        dataset_name = "training" if is_train else "validation"
        print(f"\nEvaluating on {dataset_name} dataset")
        
        # Load annotations for the selected dataset
        self.coco_gt_data, self.categories = self.load_coco_annotations(is_train=is_train)
        
        # Print information about class mapping
        if self.yolo_to_coco_mapping:
            print("\nClass mapping from YOLO to COCO (showing first 10):")
            for i, (yolo_id, coco_id) in enumerate(list(self.yolo_to_coco_mapping.items())[:10]):
                yolo_name = self.model.names[yolo_id]
                coco_name = self.categories[coco_id]
                print(f"  YOLO {yolo_id} ({yolo_name}) -> COCO {coco_id} ({coco_name})")
        
        # Determine which images to evaluate
        image_files = []
        images_dir = self.train_images_dir if is_train else self.val_images_dir
        
        if self.eval_full_dataset:
            # Get all image files from the dataset
            # This would be better done using the COCO annotations to avoid file system operations
            print(f"Getting all images from the {dataset_name} dataset...")
            for img in self.coco_gt_data['images']:
                image_files.append((img['file_name'], os.path.join(images_dir, img['file_name'])))
        else:
            # Use specific images
            for img_file in self.specific_images:
                image_path = os.path.join(images_dir, img_file)
                if os.path.exists(image_path):
                    image_files.append((img_file, image_path))
                else:
                    print(f"Warning: Image not found: {image_path}")
        
        # Limit the number of images if needed
        if self.max_images is not None and len(image_files) > self.max_images:
            print(f"Limiting evaluation to {self.max_images} images (out of {len(image_files)} total)")
            image_files = image_files[:self.max_images]
            
        # For validation dataset, create a split for calibration set if caching is enabled
        if self.enable_cache and not is_train:
            # Split validation data into validation and calibration
            val_files, cal_files = self.split_for_calibration(image_files, self.coco_gt_data)
            
            # Process validation set
            print(f"\nProcessing validation set ({len(val_files)} images)...")
            self._process_image_set(val_files, "validation")
            
            # Process calibration set
            print(f"\nProcessing calibration set ({len(cal_files)} images)...")
            self._process_image_set(cal_files, "calibration")
            
            # Return combined metrics (average of both sets)
            return None
        else:
            # Regular processing for training set or when caching is disabled
            print(f"Evaluating {len(image_files)} images...")
            return self._process_image_set(image_files, dataset_name)
            
    def _process_image_set(self, image_files, dataset_name):
        """
        Process a set of images and store results.
        
        Args:
            image_files: List of (filename, path) tuples for images
            dataset_name: Name of the dataset (e.g., "training", "validation", "calibration")
        
        Returns:
            Overall metrics for the processed images
        """
        # Try to load cache if enabled
        if self.enable_cache and self.cache:
            cache_loaded = self.cache.load_cache(dataset_name)
            if cache_loaded:
                print(f"Loaded prediction cache for {dataset_name} dataset")
                print(f"Cache contains {len(self.cache.prediction_cache)} predictions")
                if self.cache_logits and hasattr(self.cache, 'logits_cache') and self.cache.logits_cache:
                    print(f"Cache contains logits for {len(self.cache.logits_cache)} images")
            else:
                print(f"No existing cache found for {dataset_name} dataset, will create new cache")
        
        # Initialize overall results
        all_true_positives = []
        all_false_positives = []
        all_false_negatives = []
        
        # Store all ground truth in a consolidated dictionary for caching
        consolidated_ground_truth = {}
        
        # Evaluate each image
        for img_file, img_path in tqdm(image_files, desc=f"Evaluating {dataset_name} images"):
            # Get ground truth for this image
            ground_truth, image_size = self.get_image_ground_truth(img_file, self.coco_gt_data)
            
            # Store ground truth for consolidated saving
            if self.enable_cache:
                consolidated_ground_truth[img_file] = {
                    'ground_truth': ground_truth,
                    'image_size': image_size
                }
            
            if not ground_truth:
                print(f"Warning: No ground truth annotations found for {img_file}")
                continue
            
            # Evaluate the image
            result = self.evaluate_image(img_path, ground_truth, image_size)
            
            # Collect results for overall metrics
            all_true_positives.extend(result['matches']['true_positives'])
            all_false_positives.extend(result['matches']['false_positives'])
            all_false_negatives.extend(result['matches']['false_negatives'])
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(all_true_positives, all_false_positives, all_false_negatives)
        
        # Print summary
        print(f"\n===== {dataset_name.title()} Evaluation Results =====")
        print(f"Total images evaluated: {len(image_files)}")
        print(f"True positives: {overall_metrics['overall']['true_positives']}")
        print(f"False positives: {overall_metrics['overall']['false_positives']}")
        print(f"False negatives: {overall_metrics['overall']['false_negatives']}")
        print(f"Precision: {overall_metrics['overall']['precision']:.4f}")
        print(f"Recall: {overall_metrics['overall']['recall']:.4f}")
        print(f"F1 Score: {overall_metrics['overall']['f1_score']:.4f}")
        print(f"Average IoU: {overall_metrics['overall']['average_iou']:.4f}")
        
        # Print mAP metrics
        map_metrics = overall_metrics['overall']['map_metrics']
        print(f"\n===== {dataset_name.title()} mAP Metrics =====")
        print(f"mAP@50: {map_metrics['map50']:.4f}")
        print(f"mAP@75: {map_metrics['map75']:.4f}")
        print(f"mAP@50-95: {map_metrics['map50-95']:.4f}")
        
        # Print per-class metrics
        print(f"\n===== {dataset_name.title()} Per-Class Results =====")
        sorted_classes = sorted(overall_metrics['per_class'].items(), key=lambda x: x[1]['f1'], reverse=True)
        for class_name, metrics in sorted_classes:
            print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save metrics to CSV if enabled
        if self.config['metrics']['save_csv']:
            metrics_dir = os.path.join(self.metrics_dir, dataset_name)
            os.makedirs(metrics_dir, exist_ok=True)
            self.save_metrics_to_csv(overall_metrics, metrics_dir)
        
        # Save cache if enabled and modified
        if self.enable_cache and self.cache and hasattr(self.cache, 'cache_modified') and self.cache.cache_modified:
            # Save the consolidated ground truth file
            cache_path = self._get_cache_path(dataset_name)
            ground_truth_path = os.path.join(cache_path, "yolo11_ground_truth.pt")
            os.makedirs(os.path.dirname(ground_truth_path), exist_ok=True)
            torch.save(consolidated_ground_truth, ground_truth_path)
            print(f"Saved consolidated ground truth for {len(consolidated_ground_truth)} images")
            
            # Save the cache
            cache_path = self.cache.save_cache(dataset_name)
            print(f"Saved prediction cache to {cache_path}")
            if self.cache_logits and hasattr(self.cache, 'logits_cache') and self.cache.logits_cache:
                print(f"Saved logits for {len(self.cache.logits_cache)} images")
        
        return overall_metrics
    
    def _get_cache_path(self, dataset_name: str) -> str:
        """Get the path to the cache directory for a specific dataset."""
        if self.cache:
            return self.cache._get_cache_path(dataset_name)
        else:
            cache_dir = self.config.get('cache', {}).get('dir', 'yolo11_cache')
            from yolo11_cache import YOLO11Cache
            temp_cache = YOLO11Cache(cache_dir=cache_dir)
            return temp_cache._get_cache_path(dataset_name)

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model against COCO ground truth")
    parser.add_argument("--config", type=str, default="/ssd_4TB/divake/Vision_LSF/yolo11/yolo_eval_config.yaml", help="Path to config file")
    parser.add_argument("--enable-cache", action="store_true", help="Enable caching of model predictions")
    parser.add_argument("--cache-logits", action="store_true", help="Cache raw model outputs (logits) for conformal prediction")
    args = parser.parse_args()
    
    # Enable caching by default for better performance
    enable_cache = True
    cache_logits = True
    
    # If explicitly specified via args, use those values
    if args.enable_cache:
        enable_cache = args.enable_cache
    if args.cache_logits:
        cache_logits = args.cache_logits
        
    print(f"Caching is {'enabled' if enable_cache else 'disabled'}")
    print(f"Logit caching is {'enabled' if cache_logits else 'disabled'}")
    
    evaluator = COCOEvaluator(args.config, enable_cache=enable_cache, cache_logits=cache_logits)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main() 