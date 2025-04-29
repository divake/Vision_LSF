#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO11 Model Output Cache

This module provides caching functionality for YOLO11 model predictions
to avoid repeated processing of the same images.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np

class YOLO11Cache:
    """Class for caching YOLO11 model predictions."""
    
    def __init__(
        self,
        cache_dir: str = "yolo11_cache",
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        cache_logits: bool = False,
    ):
        """
        Initialize the YOLO11 cache.
        
        Args:
            cache_dir: Directory to store cached outputs
            model_name: Name of the YOLO model
            conf_threshold: Confidence threshold used for detection
            iou_threshold: IOU threshold used for NMS
            cache_logits: Whether to cache raw model outputs (logits) for conformal prediction
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cache_logits = cache_logits
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache identifier based on model and parameters
        self.cache_id = self._generate_cache_id()
        
        # Flag to track if data is cached
        self.is_cached = False
        self.cache_metadata = {}
        self.prediction_cache = {}
        self.logits_cache = {} if cache_logits else None
    
    def _generate_cache_id(self) -> str:
        """
        Generate a unique identifier for the cache based on model and parameters.
        
        Returns:
            A unique ID string for the cache
        """
        # Create a string with all parameters that affect the cache
        param_str = (
            f"model={self.model_name},"
            f"conf={self.conf_threshold},"
            f"iou={self.iou_threshold},"
            f"logits={self.cache_logits}"
        )
        
        # Generate hash of the parameter string
        cache_id = hashlib.md5(param_str.encode()).hexdigest()
        return cache_id
    
    def _get_cache_path(self, dataset_name: str) -> str:
        """
        Generate path to cache directory for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to cache directory
        """
        # For validation, don't use cache_id to simplify paths
        if dataset_name.lower() in ["validation", "val"]:
            cache_path = os.path.join(self.cache_dir, "validation")
        elif dataset_name.lower() in ["calibration", "cal"]:
            cache_path = os.path.join(self.cache_dir, "calibration")
        else:
            # For training and other datasets, still use cache_id for potential multiple configurations
            cache_path = os.path.join(self.cache_dir, f"{dataset_name}_{self.cache_id}")
        
        return cache_path
    
    def save_prediction(
        self, 
        image_path: str, 
        prediction: Dict,
        logits: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Save a model prediction to the cache.
        
        Args:
            image_path: Path to the image file
            prediction: Dictionary containing model prediction data (processed detections)
            logits: Dictionary of raw model outputs before post-processing (optional)
        """
        # Add prediction to in-memory cache
        self.prediction_cache[image_path] = prediction
        
        # If caching logits and logits are provided, store them too
        if self.cache_logits and logits is not None:
            if self.logits_cache is None:
                self.logits_cache = {}
            self.logits_cache[image_path] = logits
        
        # Mark cache as modified
        self.cache_modified = True
    
    def get_prediction(
        self, 
        image_path: str, 
        include_logits: bool = False
    ) -> Union[Dict, Tuple[Dict, Optional[Dict]]]:
        """
        Get a cached prediction for an image.
        
        Args:
            image_path: Path to the image file
            include_logits: Whether to include logits in the return value
            
        Returns:
            Cached prediction or (prediction, logits) tuple if include_logits=True
        """
        prediction = self.prediction_cache.get(image_path, None)
        
        if include_logits and self.cache_logits and self.logits_cache is not None:
            logits = self.logits_cache.get(image_path, None)
            return prediction, logits
        
        return prediction
    
    def has_prediction(self, image_path: str) -> bool:
        """
        Check if a prediction is cached for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if prediction is cached, False otherwise
        """
        return image_path in self.prediction_cache
    
    def has_logits(self, image_path: str) -> bool:
        """
        Check if logits are cached for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if logits are cached, False otherwise
        """
        if not self.cache_logits or self.logits_cache is None:
            return False
        return image_path in self.logits_cache
    
    def save_cache(self, dataset_name: str) -> str:
        """
        Save the cache to disk, consolidating data into single files.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the cache directory
        """
        # Generate cache path
        cache_path = self._get_cache_path(dataset_name)
        os.makedirs(cache_path, exist_ok=True)
        
        logging.info(f"Saving cache to {cache_path}")
        
        # Save consolidated predictions
        predictions_path = os.path.join(cache_path, "yolo11_predictions.pt")
        torch.save(self.prediction_cache, predictions_path)
        logging.info(f"Saved consolidated predictions for {len(self.prediction_cache)} images")
        
        # Save logits if enabled and available as consolidated files
        if self.cache_logits and self.logits_cache:
            # Group logits by tensor type
            probs_dict = {}
            targets_dict = {}
            
            # Separate logits into probs and targets
            for img_path, logits_dict in self.logits_cache.items():
                img_name = os.path.basename(img_path)
                
                # Store model outputs (probabilities)
                if 'probs' in logits_dict or 'conf' in logits_dict or 'cls' in logits_dict:
                    probs_dict[img_name] = {}
                    for key in ['probs', 'conf', 'cls']:
                        if key in logits_dict:
                            probs_dict[img_name][key] = logits_dict[key]
                
                # Store any ground truth/target related tensors
                # Assuming targets might be in logits with specific keys
                # This would depend on your specific implementation
                targets = {}
                for key in logits_dict:
                    if 'target' in key or 'gt' in key:
                        targets[key] = logits_dict[key]
                
                if targets:
                    targets_dict[img_name] = targets
            
            # Save consolidated probs file
            probs_path = os.path.join(cache_path, "yolo11_probs.pt")
            torch.save(probs_dict, probs_path)
            logging.info(f"Saved consolidated probs for {len(probs_dict)} images")
            
            # Save consolidated targets file if any targets were collected
            if targets_dict:
                targets_path = os.path.join(cache_path, "yolo11_targets.pt")
                torch.save(targets_dict, targets_path)
                logging.info(f"Saved consolidated targets for {len(targets_dict)} images")
            
            # For backward compatibility, also save individual tensors by type
            all_tensor_types = set()
            for logits_dict in self.logits_cache.values():
                all_tensor_types.update(logits_dict.keys())
            
            for tensor_type in all_tensor_types:
                tensors_by_img = {}
                for img_path, logits_dict in self.logits_cache.items():
                    if tensor_type in logits_dict:
                        img_name = os.path.basename(img_path)
                        tensors_by_img[img_name] = logits_dict[tensor_type]
                
                if tensors_by_img:
                    tensor_path = os.path.join(cache_path, f"yolo11_{tensor_type}.pt")
                    torch.save(tensors_by_img, tensor_path)
            
            logging.info(f"Saved consolidated tensor files for {len(all_tensor_types)} tensor types")
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "cache_id": self.cache_id,
            "creation_time": time.time(),
            "dataset_name": dataset_name,
            "num_images": len(self.prediction_cache),
            "cache_logits": self.cache_logits,
            "logits_cached": self.cache_logits and self.logits_cache is not None and len(self.logits_cache) > 0,
            "consolidated_format": True,  # Flag to indicate the new consolidated format
            "tensor_types": list(all_tensor_types) if self.cache_logits and self.logits_cache else []
        }
        
        metadata_path = os.path.join(cache_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Store metadata
        self.cache_metadata = metadata
        self.is_cached = True
        self.cache_modified = False
        
        logging.info(f"Cache saved with {len(self.prediction_cache)} predictions")
        return cache_path
    
    def load_cache(self, dataset_name: str) -> bool:
        """
        Load cached predictions from disk.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_path = self._get_cache_path(dataset_name)
        
        if not os.path.exists(cache_path):
            logging.warning(f"Cache not found at {cache_path}")
            return False
        
        # Load metadata
        metadata_path = os.path.join(cache_path, "metadata.json")
        if not os.path.exists(metadata_path):
            logging.warning(f"Cache metadata not found at {metadata_path}")
            return False
        
        with open(metadata_path, 'r') as f:
            self.cache_metadata = json.load(f)
        
        # Check if cache is using the consolidated format
        is_consolidated = self.cache_metadata.get("consolidated_format", False)
        has_logits = self.cache_metadata.get("logits_cached", False)
        
        # Load predictions based on format
        if is_consolidated:
            # Load consolidated predictions
            predictions_path = os.path.join(cache_path, "yolo11_predictions.pt")
            if os.path.exists(predictions_path):
                self.prediction_cache = torch.load(predictions_path)
                logging.info(f"Loaded consolidated predictions for {len(self.prediction_cache)} images")
            else:
                logging.warning(f"Consolidated predictions not found at {predictions_path}")
                return False
            
            # Load consolidated logits if they were cached
            if has_logits and self.cache_logits:
                self.logits_cache = {}
                
                # Load probs file
                probs_path = os.path.join(cache_path, "yolo11_probs.pt")
                if os.path.exists(probs_path):
                    probs_dict = torch.load(probs_path)
                    
                    # Initialize logits cache with probs data
                    for img_name, img_probs in probs_dict.items():
                        self.logits_cache[img_name] = img_probs
                    
                    logging.info(f"Loaded consolidated probs for {len(probs_dict)} images")
                
                # Load targets file if it exists
                targets_path = os.path.join(cache_path, "yolo11_targets.pt")
                if os.path.exists(targets_path):
                    targets_dict = torch.load(targets_path)
                    
                    # Add targets to existing logits cache entries
                    for img_name, img_targets in targets_dict.items():
                        if img_name in self.logits_cache:
                            self.logits_cache[img_name].update(img_targets)
                        else:
                            self.logits_cache[img_name] = img_targets
                    
                    logging.info(f"Loaded consolidated targets for {len(targets_dict)} images")
                
                # Handle tensor-specific files if probs/targets are missing
                tensor_types = self.cache_metadata.get("tensor_types", [])
                for tensor_type in tensor_types:
                    tensor_path = os.path.join(cache_path, f"yolo11_{tensor_type}.pt")
                    if os.path.exists(tensor_path):
                        tensors_dict = torch.load(tensor_path)
                        
                        for img_name, tensor in tensors_dict.items():
                            if img_name not in self.logits_cache:
                                self.logits_cache[img_name] = {}
                            self.logits_cache[img_name][tensor_type] = tensor
                
                logging.info(f"Loaded logits for {len(self.logits_cache)} images")
        else:
            # Legacy format - load from individual files
            predictions_path = os.path.join(cache_path, "predictions.json")
            if not os.path.exists(predictions_path):
                logging.warning(f"Predictions not found at {predictions_path}")
                return False
            
            with open(predictions_path, 'r') as f:
                serialized_predictions = json.load(f)
            
            # Convert to usable format and store in memory
            self.prediction_cache = {}
            
            for img_name, pred in serialized_predictions.items():
                # Get boxes data
                boxes_data = []
                for box in pred.get('boxes', []):
                    # Convert lists back to numpy arrays if needed
                    processed_box = {}
                    for k, v in box.items():
                        if k in ['bbox', 'confidence', 'class_id'] and isinstance(v, list):
                            processed_box[k] = np.array(v)
                        else:
                            processed_box[k] = v
                    boxes_data.append(processed_box)
                
                self.prediction_cache[img_name] = {'boxes': boxes_data}
            
            # Load logits if they were cached
            if has_logits and self.cache_logits:
                logits_dir = os.path.join(cache_path, "logits")
                if os.path.exists(logits_dir):
                    self.logits_cache = {}
                    
                    # Find all logits files
                    logits_files = [f for f in os.listdir(logits_dir) if f.endswith('.pt')]
                    
                    # Group by image name
                    img_logits = {}
                    for logits_file in logits_files:
                        parts = logits_file.rsplit('_', 1)
                        if len(parts) == 2:
                            img_name = parts[0]
                            tensor_name = parts[1].replace('.pt', '')
                            
                            if img_name not in img_logits:
                                img_logits[img_name] = []
                            
                            img_logits[img_name].append((tensor_name, os.path.join(logits_dir, logits_file)))
                    
                    # Load tensors for each image
                    for img_name, tensor_files in img_logits.items():
                        self.logits_cache[img_name] = {}
                        for tensor_name, tensor_path in tensor_files:
                            self.logits_cache[img_name][tensor_name] = torch.load(tensor_path)
                    
                    logging.info(f"Loaded logits for {len(self.logits_cache)} images")
                else:
                    logging.warning(f"Logits directory not found at {logits_dir}")
        
        self.is_cached = True
        self.cache_modified = False
        
        logging.info(f"Loaded cache from {cache_path} with {len(self.prediction_cache)} predictions")
        logging.info(f"Cache created on: {time.ctime(self.cache_metadata['creation_time'])}")
        
        return True
    
    def clear_cache(self) -> None:
        """Clear the cache from memory."""
        self.prediction_cache = {}
        if self.cache_logits:
            self.logits_cache = {}
        self.is_cached = False
        self.cache_modified = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = {
            "num_cached_predictions": len(self.prediction_cache),
            "is_cached": self.is_cached,
            "cache_id": self.cache_id,
            "model_name": self.model_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "cache_logits": self.cache_logits,
            "metadata": self.cache_metadata
        }
        
        if self.cache_logits and self.logits_cache is not None:
            stats["num_cached_logits"] = len(self.logits_cache)
        
        return stats 