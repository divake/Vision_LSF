#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COCO Dataset and YOLO Model Validator

This script performs diagnostic checks on the COCO dataset structure and YOLO model
to help identify why the evaluation metrics might be zero.
"""

import os
import json
import yaml
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class DatasetValidator:
    """Class for validating COCO dataset and YOLO model compatibility."""
    
    def __init__(self, config_path):
        """Initialize the validator with config file path."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract paths
        self.train_images_dir = self.config['paths']['coco_dataset']['train_images']
        self.val_images_dir = self.config['paths']['coco_dataset']['val_images']
        self.train_annotations_file = self.config['paths']['coco_dataset']['train_annotations']
        self.val_annotations_file = self.config['paths']['coco_dataset']['val_annotations']
        self.model_name = self.config['model']['name']
        
        # For storing loaded data
        self.coco_data = None
        self.categories = None
        self.model = None
    
    def check_paths_exist(self):
        """Check if all required paths exist."""
        print("\n=== Checking paths existence ===")
        paths_to_check = {
            "Train images directory": self.train_images_dir,
            "Validation images directory": self.val_images_dir,
            "Train annotations file": self.train_annotations_file,
            "Validation annotations file": self.val_annotations_file,
        }
        
        all_exist = True
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            print(f"{name}: {path} - {'EXISTS' if exists else 'MISSING'}")
            if not exists:
                all_exist = False
        
        # Check for model file
        model_path = os.path.join('/ssd_4TB/divake/Vision_LSF/yolo11', self.model_name)
        model_exists = os.path.exists(model_path)
        print(f"Model file: {model_path} - {'EXISTS' if model_exists else 'MISSING'}")
        
        if not all_exist:
            print("\n⚠️ Some required paths are missing. Please fix before proceeding.")
        else:
            print("\n✅ All required paths exist.")
        
        return all_exist
    
    def load_annotations(self, is_train=True):
        """Load and validate COCO annotations."""
        print("\n=== Loading and validating annotations ===")
        annotation_file = self.train_annotations_file if is_train else self.val_annotations_file
        print(f"Loading annotations from {annotation_file}")
        
        try:
            with open(annotation_file, 'r') as f:
                self.coco_data = json.load(f)
            
            # Check required fields
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in self.coco_data:
                    print(f"⚠️ Missing required field '{field}' in annotations file")
                    return False
            
            # Create category mapping
            self.categories = {}
            for cat in self.coco_data['categories']:
                self.categories[cat['id']] = cat['name']
            
            # Print statistics
            print(f"- Images count: {len(self.coco_data['images'])}")
            print(f"- Annotations count: {len(self.coco_data['annotations'])}")
            print(f"- Categories count: {len(self.categories)}")
            
            # Validate a few random annotations
            print("\nSample categories:")
            for i, (cat_id, cat_name) in enumerate(list(self.categories.items())[:10]):
                print(f"  {cat_id}: {cat_name}")
            
            print("\n✅ Annotations loaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ Error loading annotations: {e}")
            return False
    
    def check_specific_images(self):
        """Check if specific images from config exist and have annotations."""
        if not self.config['evaluation']['specific_images']:
            print("\n⚠️ No specific images defined in config.")
            return False
        
        print("\n=== Checking specific images ===")
        all_valid = True
        
        for img_file in self.config['evaluation']['specific_images']:
            # Check if image exists
            img_path = os.path.join(self.train_images_dir, img_file)
            img_exists = os.path.exists(img_path)
            
            # Find image and its annotations in COCO data
            image_id = None
            for img in self.coco_data['images']:
                if img['file_name'] == img_file:
                    image_id = img['id']
                    break
            
            if image_id is None:
                print(f"⚠️ Image {img_file} not found in COCO annotations")
                all_valid = False
                continue
            
            # Count annotations for this image
            annotation_count = 0
            for ann in self.coco_data['annotations']:
                if ann['image_id'] == image_id:
                    annotation_count += 1
            
            print(f"Image: {img_file}")
            print(f"  - File exists: {'✅' if img_exists else '❌'}")
            print(f"  - Found in annotations: {'✅' if image_id is not None else '❌'}")
            print(f"  - Annotation count: {annotation_count}")
            
            if annotation_count == 0:
                print(f"  ⚠️ Warning: No annotations found for this image")
                all_valid = False
        
        if all_valid:
            print("\n✅ All specific images exist and have annotations")
        else:
            print("\n⚠️ Some issues found with specific images")
            
        return all_valid
    
    def load_model(self):
        """Load YOLO model and check if it works."""
        print("\n=== Loading and testing YOLO model ===")
        
        try:
            # Check if model exists locally first
            model_path = os.path.join('/ssd_4TB/divake/Vision_LSF/yolo11', self.model_name)
            if os.path.exists(model_path):
                print(f"Using existing model at {model_path}")
                self.model = YOLO(model_path)
            else:
                print(f"Model not found at {model_path}, using default path or downloading")
                self.model = YOLO(self.model_name)
            
            print("Model loaded successfully")
            
            # Print model classes
            print("\nModel categories:")
            for i, (class_id, class_name) in enumerate(self.model.names.items()):
                if i < 10:  # Only show first 10 classes
                    print(f"  {class_id}: {class_name}")
            
            return True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return False
    
    def test_sample_prediction(self):
        """Test model prediction on a sample image."""
        if not self.model or not self.coco_data:
            print("⚠️ Model or annotations not loaded. Run load_model() and load_annotations() first.")
            return False
        
        print("\n=== Testing sample prediction ===")
        
        # Use the first specific image from config
        if self.config['evaluation']['specific_images']:
            img_file = self.config['evaluation']['specific_images'][0]
            img_path = os.path.join(self.train_images_dir, img_file)
            
            if not os.path.exists(img_path):
                print(f"⚠️ Sample image not found: {img_path}")
                return False
            
            print(f"Running prediction on {img_path}")
            
            # Run prediction
            results = self.model(
                img_path, 
                conf=self.config['model']['conf_threshold'], 
                iou=self.config['model']['iou_threshold'],
                verbose=False
            )
            
            # Display results
            result = results[0]
            boxes = result.boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
            
            print(f"Detected {len(boxes)} objects")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                cls_name = self.model.names[cls_id]
                print(f"  {i+1}: {cls_name}, confidence: {conf:.2f}, bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # Get ground truth for this image
            image_id = None
            for img in self.coco_data['images']:
                if img['file_name'] == img_file:
                    image_id = img['id']
                    break
            
            if image_id is None:
                print(f"⚠️ Image not found in COCO annotations")
                return False
            
            # Get annotations for this image
            gt_objects = []
            for ann in self.coco_data['annotations']:
                if ann['image_id'] == image_id:
                    x, y, w, h = ann['bbox']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    category_id = ann['category_id']
                    category_name = self.categories.get(category_id, 'unknown')
                    gt_objects.append({
                        'bbox': [x1, y1, x2, y2],
                        'category_id': category_id,
                        'category_name': category_name
                    })
            
            print(f"\nGround truth: {len(gt_objects)} objects")
            for i, gt in enumerate(gt_objects):
                print(f"  {i+1}: {gt['category_name']}, bbox: {gt['bbox']}")
            
            # Check class mapping between YOLO and COCO
            print("\nAnalyzing class mapping between YOLO and COCO categories:")
            
            # Find overlapping categories
            yolo_classes = set(self.model.names.values())
            coco_classes = set(self.categories.values())
            
            overlapping = yolo_classes.intersection(coco_classes)
            print(f"Overlapping categories: {len(overlapping)} out of {len(coco_classes)} COCO categories")
            
            if len(overlapping) < 10:
                print("Sample overlapping categories:")
                for cat in list(overlapping)[:10]:
                    print(f"  - {cat}")
            
            # Check for potential class mapping issues
            if len(overlapping) < len(coco_classes):
                print("\n⚠️ Potential class mapping issue detected:")
                print("  The YOLO model and COCO dataset don't share all category names.")
                print("  This could cause evaluation metrics to be zero if classes don't match.")
                
                # Find COCO classes not in YOLO
                coco_only = coco_classes - yolo_classes
                if coco_only:
                    print("\n  COCO categories not found in YOLO model:")
                    for cat in list(coco_only)[:10]:
                        print(f"    - {cat}")
                
                # Find YOLO classes not in COCO
                yolo_only = yolo_classes - coco_classes
                if yolo_only:
                    print("\n  YOLO categories not found in COCO dataset:")
                    for cat in list(yolo_only)[:10]:
                        print(f"    - {cat}")
            
            # Visual test with matplotlib
            try:
                print("\nCreating visualization for comparison...")
                
                # Load image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create figure
                plt.figure(figsize=(12, 8))
                plt.imshow(image)
                
                # Draw ground truth boxes in green
                for gt in gt_objects:
                    bbox = gt['bbox']
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, edgecolor='green', linewidth=2))
                    plt.text(x1, y1-10, f"GT: {gt['category_name']}", 
                           color='green', fontsize=10, backgroundcolor='white')
                
                # Draw prediction boxes in blue
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    x1, y1, x2, y2 = [int(c) for c in [x1, y1, x2, y2]]
                    cls_id = int(cls_id)
                    cls_name = self.model.names[cls_id]
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, edgecolor='blue', linewidth=2))
                    plt.text(x1, y1-10, f"Pred: {cls_name} {conf:.2f}", 
                           color='blue', fontsize=10, backgroundcolor='white')
                
                # Save the image
                vis_path = os.path.join('/ssd_4TB/divake/Vision_LSF/yolo11', 
                                      os.path.basename(img_path).split('.')[0] + '_debug.jpg')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(vis_path)
                print(f"Visualization saved to {vis_path}")
            except Exception as e:
                print(f"⚠️ Error creating visualization: {e}")
            
            return True
        else:
            print("⚠️ No specific images defined in config")
            return False
    
    def check_class_mapping(self):
        """Analyze class mapping between YOLO model and COCO dataset."""
        print("\n=== Analyzing YOLO-to-COCO class mapping in detail ===")
        
        if not self.model or not self.categories:
            print("⚠️ Model or annotations not loaded")
            return False
        
        # Print statistics
        print(f"YOLO model has {len(self.model.names)} classes")
        print(f"COCO dataset has {len(self.categories)} categories")
        
        # Create reverse mapping of category names to IDs for both
        yolo_name_to_id = {name.lower(): id for id, name in self.model.names.items()}
        coco_name_to_id = {name.lower(): id for id, name in self.categories.items()}
        
        # Find direct matches
        direct_matches = []
        for coco_id, coco_name in self.categories.items():
            yolo_name_lower = coco_name.lower()
            if yolo_name_lower in yolo_name_to_id:
                yolo_id = yolo_name_to_id[yolo_name_lower]
                direct_matches.append((coco_id, coco_name, yolo_id, self.model.names[yolo_id]))
        
        print(f"\nDirect matches: {len(direct_matches)} out of {len(self.categories)} categories")
        
        # Print some matches
        if direct_matches:
            print("\nSample direct matches (COCO ID, COCO Name, YOLO ID, YOLO Name):")
            for match in direct_matches[:10]:
                print(f"  {match[0]}: {match[1]} <-> {match[2]}: {match[3]}")
        
        # Print all YOLO classes for reference
        print("\nAll YOLO classes (first 20):")
        sorted_yolo = sorted(self.model.names.items(), key=lambda x: x[0])
        for i, (cls_id, cls_name) in enumerate(sorted_yolo):
            if i < 20:
                print(f"  {cls_id}: {cls_name}")
        
        # Print all COCO categories for reference
        print("\nAll COCO categories (first 20):")
        sorted_coco = sorted(self.categories.items(), key=lambda x: x[0])
        for i, (cat_id, cat_name) in enumerate(sorted_coco):
            if i < 20:
                print(f"  {cat_id}: {cat_name}")
        
        if len(direct_matches) < len(self.categories) * 0.8:
            print("\n⚠️ Less than 80% of COCO categories match YOLO classes directly.")
            print("  This could be why evaluation metrics are zero.")
            print("  Here are some suggested solutions:")
            print("  1. Create a mapping between COCO categories and YOLO classes")
            print("  2. Use a YOLO model trained specifically on COCO")
            print("  3. Modify the evaluation code to handle class mismatches")
        else:
            print("\n✅ Most COCO categories have direct matches with YOLO classes")
        
        return True
    
    def run_all_checks(self):
        """Run all validation checks."""
        print("=== Running all dataset and model validation checks ===")
        
        checks_passed = True
        
        # Check paths
        if not self.check_paths_exist():
            checks_passed = False
        
        # Load annotations
        if not self.load_annotations():
            checks_passed = False
        
        # Check specific images
        if not self.check_specific_images():
            checks_passed = False
        
        # Load model
        if not self.load_model():
            checks_passed = False
        
        # Check class mapping
        if not self.check_class_mapping():
            checks_passed = False
        
        # Test sample prediction
        if not self.test_sample_prediction():
            checks_passed = False
        
        # Summary
        print("\n=== Validation Summary ===")
        if checks_passed:
            print("✅ All checks passed!")
        else:
            print("⚠️ Some checks failed. Please review the issues above.")
        
        return checks_passed

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate COCO dataset and YOLO model")
    parser.add_argument("--config", type=str, default="/ssd_4TB/divake/Vision_LSF/yolo11/yolo_eval_config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    validator = DatasetValidator(args.config)
    validator.run_all_checks()

if __name__ == "__main__":
    main() 