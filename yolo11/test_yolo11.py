# Save this file as test_yolo11.py in the yolo11 directory

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import torch
import cv2
import matplotlib.pyplot as plt
from yolo11_feature_extractor import YOLO11FeatureExtractor

def process_single_image(image_path, model_name="yolo11n.pt", layer_name=None, conf=0.25, 
                         device=None, output_dir="yolo11_output", show_plot=True):
    """Process a single image with the YOLO11 model"""
    
    # Check if the image exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return
    
    # Initialize the feature extractor
    print(f"Loading model {model_name}...")
    try:
        extractor = YOLO11FeatureExtractor(
            model_name=model_name,
            device=device,
            conf_threshold=conf
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert layer_name to list format if specified
    layer_names = None
    if layer_name:
        if isinstance(layer_name, str):
            layer_names = [name.strip() for name in layer_name.split(',')]
        else:
            layer_names = layer_name
    
    # Run prediction directly
    print(f"Processing image: {image_path}")
    results = extractor.model(image_path, conf=conf, verbose=False)
    
    # Get the result for visualization
    result = results[0]
    
    # Extract detections
    boxes = result.boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
    
    # Convert detections to a more readable format
    detections = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(conf),
            'class_id': cls_id,
            'class_name': extractor.class_names[cls_id]
        })
    
    # Display results
    print("\nDetection Results:")
    print(f"Detected {len(detections)} objects:")
    
    for i, det in enumerate(detections):
        class_name = det['class_name']
        confidence = det['confidence']
        bbox = det['bbox']
        print(f"  {i+1}. {class_name} ({confidence:.2f}): bbox={[round(b, 1) for b in bbox]}")
    
    # Get visualization
    vis_img = result.plot()
    
    # Save visualization
    vis_path = os.path.join(output_dir, "detection_result.jpg")
    cv2.imwrite(vis_path, vis_img)
    print(f"\nVisualization saved to: {vis_path}")
    
    # Extract features if requested
    if layer_names:
        # Need to reprocess with register_hooks for feature extraction
        # This creates a temporary directory structure as required by process_images
        temp_dir = os.path.join(output_dir, "temp_img")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy or link the image to the temp directory
        img_name = os.path.basename(image_path)
        img_temp_path = os.path.join(temp_dir, img_name)
        
        try:
            # Try symlink first (more efficient)
            if os.path.exists(img_temp_path):
                os.remove(img_temp_path)
            os.symlink(os.path.abspath(image_path), img_temp_path)
        except:
            # Fall back to copy
            import shutil
            shutil.copy2(image_path, img_temp_path)
        
        # Process for feature extraction
        feature_results = extractor.process_images(
            img_dir=temp_dir,
            output_dir=output_dir,
            layer_names=layer_names,
            batch_size=1,
            save_visualizations=False
        )
        
        # Print feature information
        for img_name, result in feature_results.items():
            if 'feature_shapes' in result and result['feature_shapes']:
                print("\nExtracted Features:")
                for layer, shape in result['feature_shapes'].items():
                    print(f"  Layer '{layer}': shape={shape}")
                    print(f"  Saved to: {result['feature_files'][layer]}")
        
        # Clean up temp directory
        try:
            os.remove(img_temp_path)
            os.rmdir(temp_dir)
        except:
            pass
    
    # Show plot
    if show_plot:
        rgb_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.title("YOLO11 Detection Results")
        plt.tight_layout()
        plt.show()
    
    return detections, vis_path

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test YOLO11 Feature Extractor on a single image")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO11 model to use")
    parser.add_argument("--layer", help="Layer(s) to extract features from (comma-separated)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", help="Device to run on (cuda, cpu)")
    parser.add_argument("--output", default="yolo11_output", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Don't display the plot window")
    args = parser.parse_args()
    
    process_single_image(
        image_path=args.image,
        model_name=args.model,
        layer_name=args.layer,
        conf=args.conf,
        device=args.device,
        output_dir=args.output,
        show_plot=not args.no_plot
    )