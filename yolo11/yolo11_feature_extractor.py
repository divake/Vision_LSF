#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from typing import Optional, List, Tuple, Dict, Any, Union, Set
import matplotlib.pyplot as plt


class YOLO11FeatureExtractor:
    """Class for extracting features and detections from YOLO11 models."""
    
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize the YOLO11 feature extractor.
        
        Args:
            model_name: Name of the YOLO11 model to use (yolo11n.pt, yolo11s.pt, etc.)
            device: Device to run inference on ("cuda", "cpu", etc.). If None, auto-selects.
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_name)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Store class names for later use
        self.class_names = self.model.names
    
    def _get_layer_by_name(self, layer_name: str) -> torch.nn.Module:
        """
        Get a specific layer from the model by name.
        
        Args:
            layer_name: Name of the layer to extract
            
        Returns:
            The requested layer module
        """
        # Try different approaches to get the layer since model structure can vary
        try:
            # Method 1: Direct attribute access on model
            if hasattr(self.model, layer_name):
                return getattr(self.model, layer_name)
            
            # Method 2: Access via model.model (common in YOLO)
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, layer_name):
                    return getattr(self.model.model, layer_name)
                
                # Method 3: Access via model.model.model (for nested structures)
                if hasattr(self.model.model, 'model'):
                    model_layers = self.model.model.model
                    
                    # Handle dot notation (e.g., "10.conv")
                    if '.' in layer_name:
                        parts = layer_name.split('.')
                        current = model_layers
                        
                        for part in parts:
                            if part.isdigit() and isinstance(current, (list, tuple)) and int(part) < len(current):
                                current = current[int(part)]
                            elif hasattr(current, part):
                                current = getattr(current, part)
                            else:
                                raise ValueError(f"Could not navigate to {part} in {layer_name}")
                        
                        return current
                    # Handle numeric indices (e.g., "10")
                    elif layer_name.isdigit() and isinstance(model_layers, (list, tuple)):
                        idx = int(layer_name)
                        if idx < len(model_layers):
                            return model_layers[idx]
                    # Handle direct attribute
                    elif hasattr(model_layers, layer_name):
                        return getattr(model_layers, layer_name)
            
            # If all methods failed, print model structure and raise error
            print("Model structure not as expected. Printing model summary for reference:")
            print(self.model)
            if hasattr(self.model, 'model'):
                print("\nModel.model structure:")
                print(self.model.model)
                if hasattr(self.model.model, 'model'):
                    print("\nModel.model.model structure:")
                    print(self.model.model.model)
            
            raise ValueError(f"Could not find layer: {layer_name}")
        except Exception as e:
            print(f"Error getting layer {layer_name}: {e}")
            print("Available top-level model attributes:", dir(self.model))
            if hasattr(self.model, 'model'):
                print("Available model.model attributes:", dir(self.model.model))
            raise

    def register_hooks(self, layer_names: Union[str, List[str]]) -> None:
        """
        Register forward hooks to extract features from specific layers.
        
        Args:
            layer_names: Name or list of names of layers to extract features from
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        self.features = {}
        self.hooks = []
        
        for layer_name in layer_names:
            try:
                layer = self._get_layer_by_name(layer_name)
                
                def hook_fn(name):
                    # Create closure to capture the correct layer_name
                    def fn(module, input, output):
                        self.features[name] = output.detach().cpu()
                    return fn
                
                # Register the hook with the specific layer name
                handle = layer.register_forward_hook(hook_fn(layer_name))
                self.hooks.append(handle)
                print(f"Registered hook for layer: {layer_name}")
            except Exception as e:
                print(f"Warning: Could not register hook for layer {layer_name}: {e}")

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            print("All hooks removed")

    def normalize_features(self, features: torch.Tensor, method: str = 'l2') -> torch.Tensor:
        """
        Normalize features using specified method.
        
        Args:
            features: Features tensor to normalize
            method: Normalization method ('l2', 'min_max', or 'standardize')
            
        Returns:
            Normalized features
        """
        if method == 'l2':
            # L2 normalization along feature dimension
            norm = torch.norm(features, p=2, dim=1, keepdim=True)
            return features / (norm + 1e-7)  # Add small epsilon to avoid division by zero
        elif method == 'min_max':
            # Min-max normalization
            min_val = torch.min(features)
            max_val = torch.max(features)
            return (features - min_val) / (max_val - min_val + 1e-7)
        elif method == 'standardize':
            # Standardization (zero mean, unit variance)
            mean = torch.mean(features)
            std = torch.std(features)
            return (features - mean) / (std + 1e-7)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def process_images(
        self,
        img_dir: str,
        output_dir: str,
        layer_names: Optional[Union[str, List[str]]] = None,
        batch_size: int = 4,
        save_visualizations: bool = True,
        max_images: Optional[int] = None,
        output_format: str = 'numpy',
        normalize_features: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process a directory of images and extract features and detections.
        
        Args:
            img_dir: Directory containing images to process
            output_dir: Directory to save results to
            layer_names: Name or list of names of layers to extract features from
            batch_size: Batch size for processing
            save_visualizations: Whether to save visualization images
            max_images: Maximum number of images to process
            output_format: Format to save detections ('numpy', 'json', or 'both')
            normalize_features: Method to normalize features (None, 'l2', 'min_max', 'standardize')
            
        Returns:
            Dictionary of results for each image
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        features_dir = os.path.join(output_dir, "features")
        detections_dir = os.path.join(output_dir, "detections")
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(detections_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(visualizations_dir, exist_ok=True)
        
        # Register hooks for feature extraction if layer names are provided
        if layer_names:
            self.register_hooks(layer_names)
        
        # Convert single layer name to list for consistent processing
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        # Get all image files
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if any(f.lower().endswith(ext) for ext in img_extensions)
        ]
        
        if max_images is not None and max_images > 0:
            image_files = image_files[:max_images]
        
        results = {}
        
        # Process images in batches with progress bar
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
            batch_files = image_files[i:i+batch_size]
            batch_results = self.model(
                batch_files,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device
            )
            
            # Process each result in the batch
            for j, (img_file, result) in enumerate(zip(batch_files, batch_results)):
                img_name = os.path.basename(img_file)
                base_name = os.path.splitext(img_name)[0]
                
                # Convert results to a more manageable format
                boxes = result.boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
                
                # Extract detections (bounding boxes, classes, confidence scores)
                detections = []
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    cls_id = int(cls_id)
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id]
                    })
                
                # Save detections in the specified format(s)
                detection_files = {}
                
                if output_format in ['numpy', 'both']:
                    npy_file = os.path.join(detections_dir, f"{base_name}_detections.npy")
                    np.save(npy_file, np.array(boxes))
                    detection_files['numpy'] = npy_file
                
                if output_format in ['json', 'both']:
                    json_file = os.path.join(detections_dir, f"{base_name}_detections.json")
                    with open(json_file, 'w') as f:
                        json.dump(detections, f, indent=2)
                    detection_files['json'] = json_file
                
                # Extract and save features if hooks were registered
                feature_files = {}
                feature_shapes = {}
                
                if layer_names and hasattr(self, 'features'):
                    for layer_name in self.features:
                        # Get features for this layer
                        features = self.features[layer_name]
                        
                        # Normalize features if requested
                        if normalize_features:
                            features = self.normalize_features(features, method=normalize_features)
                        
                        # Save features
                        features_file = os.path.join(features_dir, f"{base_name}_{layer_name.replace('.', '_')}_features.pt")
                        torch.save(features, features_file)
                        
                        feature_files[layer_name] = features_file
                        feature_shapes[layer_name] = list(features.shape)
                
                # Save visualization if requested
                vis_file = None
                if save_visualizations:
                    # Use the result's plot method which visualizes detections
                    fig = result.plot()
                    vis_file = os.path.join(visualizations_dir, f"{base_name}_vis.jpg")
                    cv2.imwrite(vis_file, fig)
                
                # Store the results
                results[img_name] = {
                    'detections': detections,
                    'detection_files': detection_files,
                    'feature_files': feature_files,
                    'feature_shapes': feature_shapes,
                    'visualization_file': vis_file
                }
        
        # Remove hooks after processing
        if layer_names:
            self.remove_hooks()
        
        return results
    
    def get_available_layers(self) -> List[str]:
        """
        Get a list of available layer names in the model.
        
        Returns:
            List of layer names that can be used for feature extraction
        """
        layers = []
        
        # Try to get layers from different model structures
        if hasattr(self.model.model, 'model'):
            model_layers = self.model.model.model
            
            # Handle list structure (common in YOLO)
            if isinstance(model_layers, (list, tuple)):
                for i, layer in enumerate(model_layers):
                    layers.append(str(i))
                    for name, _ in layer.named_children():
                        layers.append(f"{i}.{name}")
            
            # Handle module structure
            else:
                for name, _ in model_layers.named_children():
                    layers.append(name)
                    for subname, _ in getattr(model_layers, name).named_children():
                        layers.append(f"{name}.{subname}")
        
        # Add top-level modules as fallback
        for name, module in self.model.named_modules():
            if name and name not in layers:
                layers.append(name)
        
        return sorted(layers)


def main():
    parser = argparse.ArgumentParser(description="YOLO11 Feature Extractor")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO11 model to use")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="yolo11_output", help="Output directory")
    parser.add_argument("--layer", type=str, help="Layer to extract features from (can be comma-separated for multiple layers)")
    parser.add_argument("--list-layers", action="store_true", help="List available layers and exit")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--device", type=str, help="Device to run on (cuda, cpu)")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--output-format", type=str, default="numpy", choices=["numpy", "json", "both"], 
                        help="Format for saving detections (numpy, json, or both)")
    parser.add_argument("--normalize", type=str, choices=["l2", "min_max", "standardize"], 
                        help="Normalize features using specified method")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization saving")
    args = parser.parse_args()
    
    # Initialize the feature extractor
    extractor = YOLO11FeatureExtractor(
        model_name=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # List available layers if requested
    if args.list_layers:
        layers = extractor.get_available_layers()
        print("Available layers:")
        for layer in layers:
            print(f"  - {layer}")
        sys.exit(0)
    
    # Parse layers (support multiple layers separated by comma)
    layer_names = None
    if args.layer:
        layer_names = [layer.strip() for layer in args.layer.split(',')]
    
    # Process images
    results = extractor.process_images(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        layer_names=layer_names,
        batch_size=args.batch_size,
        save_visualizations=not args.no_viz,
        max_images=args.max_images,
        output_format=args.output_format,
        normalize_features=args.normalize
    )
    
    # Print summary
    print(f"Processed {len(results)} images")
    if layer_names:
        for layer in layer_names:
            # Only report on layers that were successfully extracted
            successfully_extracted = any(layer in result['feature_shapes'] for result in results.values())
            if successfully_extracted:
                print(f"Extracted features from layer '{layer}'")
    print(f"Results saved to {args.output_dir}")
    
    # Check if we have visualization files before attempting to display them
    if not args.no_viz:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        if os.path.exists(vis_dir):
            vis_files = [os.path.join(vis_dir, f) for f in os.listdir(vis_dir) 
                         if f.endswith('_vis.jpg')][:4]  # Show up to 4 examples
            
            if vis_files:
                try:
                    fig, axes = plt.subplots(1, len(vis_files), figsize=(16, 4))
                    if len(vis_files) == 1:
                        axes = [axes]
                    
                    for ax, vis_file in zip(axes, vis_files):
                        img = cv2.imread(vis_file)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                        ax.set_title(os.path.basename(vis_file))
                        ax.axis('off')
                    
                    plt.tight_layout()
                    examples_file = os.path.join(args.output_dir, "examples.png")
                    plt.savefig(examples_file)
                    print(f"Example visualizations saved to {examples_file}")
                except Exception as e:
                    print(f"Warning: Could not create visualization summary: {e}")


if __name__ == "__main__":
    main() 