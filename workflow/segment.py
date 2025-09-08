import os.path
from os.path import split
from collections import Counter
from pathlib import Path
import torch

from .node import BaseNode
# The dependency is now back inside the node, as requested.
from modules.GroundedSam4FACT.predict import load_sam2, load_dino, predict_mask

import cv2
import numpy as np
from PIL import Image

# Constants for the thresholding logic
upper_threshold = 0.9
lower_threshold = 0.1

def mask_rate(mask):
    """Calculates the ratio of non-zero pixels in a mask."""
    total_pixels = mask.size
    non_zero_pixels = np.count_nonzero(mask)
    rate = non_zero_pixels / total_pixels
    print(f"Mask rate: {rate:.4f}")
    return rate

def save_masks_as_files(masks: np.ndarray, label: str, output_path: str):
    """Saves a boolean or uint8 mask to a file."""
    if masks.dtype != np.uint8:
        mask_np = masks.astype(np.uint8) * 255
    else:
        mask_np = masks
    
    base, _ = os.path.splitext(output_path)
    save_path = f'{base}_{label}.png'
    
    mask_image = Image.fromarray(mask_np)
    mask_image.save(save_path)

def auto_resize(image_processor, max_pixels=2500000):
    """Resizes an image if it exceeds a maximum pixel count, returns the new path."""
    img = image_processor.img_data
    if img.size[0] * img.size[1] <= max_pixels:
        return image_processor.img_path, False # No resize needed

    print(f"Resizing {image_processor.name} as it exceeds {max_pixels} pixels.")
    resize_scale = np.sqrt(max_pixels / (img.size[0] * img.size[1]))
    new_size = (int(img.size[0] * resize_scale), int(img.size[1] * resize_scale))
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    temp_dir = Path(image_processor.building_obj.temp_path)
    temp_dir.mkdir(exist_ok=True)
    resized_path = temp_dir / image_processor.name
    resized_img.save(resized_path, "PNG")
    return str(resized_path), True

def analyse_mask(history, prompt, sam_predictor, dino_predictor, img_path):
    """
    Analyzes a history of mask predictions to find the optimal one.
    
    This is a simplified implementation. It selects the mask generated at the
    median threshold, which provides a basic heuristic for a stable result.
    The original complex logic was not complete.
    """
    masks = None
    infered_missing = None
    
    if not history:
        return None, None

    # Simple heuristic: return the mask from the middle of the threshold range.
    # This avoids extremes where the mask might be empty or cover the whole image.
    median_index = len(history) // 2
    masks = history[median_index]['mask']
    
    # The logic for inferring missing parts is not implemented in this version.
    infered_missing = None

    return masks, infered_missing

class Segment(BaseNode):
    """A self-contained node for segmenting images using GroundedSAM."""
    def __init__(self, image_processors):
        super().__init__(image_processors)
        # Models are loaded here, during the node's instantiation.
        print("Loading GroundedSAM models for Segment node...")
        self.sam_predictor = load_sam2()
        self.dino_predictor = load_dino()
        print("GroundedSAM models loaded.")

    def process(self, prompts=["window", "wall", "door"]):
        """Processes each image, segmenting it based on the provided prompts."""
        for img_processor in self.image_processors:
            print(f"--- Segmenting {img_processor.name} ---")
            
            img_path_for_prediction, resized = auto_resize(img_processor)

            for prompt in prompts:
                print(f"  - Prompt: '{prompt}'")
                predict_history = []
                mask_path = os.path.join(img_processor.building_obj.temp_path, img_processor.name)

                # Iterative prediction loop
                box_threshold = lower_threshold
                while box_threshold <= upper_threshold:
                    current_masks = predict_mask(
                        prompt, 
                        self.sam_predictor, 
                        self.dino_predictor, 
                        img_path_for_prediction, 
                        box_threshold, 
                        box_threshold # text_threshold is the same
                    )
                    
                    rate = mask_rate(current_masks)
                    predict_history.append({
                        'mask': current_masks,
                        'rate': rate,
                        'threshold': box_threshold
                    })
                    if rate == 0.0 and box_threshold > 0.5:
                        break # Stop if no masks are found at a high threshold
                    box_threshold += 0.05

                # Analyze the results to get the best mask
                final_mask, missing_mask = analyse_mask(
                    predict_history, 
                    prompt, 
                    self.sam_predictor, 
                    self.dino_predictor, 
                    img_path_for_prediction
                )

                if final_mask is None:
                    print(f"    No suitable mask found for prompt '{prompt}'.")
                    continue

                # Resize mask back if original image was resized
                if resized:
                    final_mask = cv2.resize(
                        (final_mask > 0).astype(np.uint8) * 255, 
                        img_processor.img_data.size, 
                        interpolation=cv2.INTER_NEAREST
                    ) > 0

                save_masks_as_files(final_mask, prompt, mask_path)

                if missing_mask is not None:
                    if resized:
                        missing_mask = cv2.resize(
                            (missing_mask > 0).astype(np.uint8) * 255, 
                            img_processor.img_data.size, 
                            interpolation=cv2.INTER_NEAREST
                        ) > 0
                    save_masks_as_files(missing_mask, f"{prompt}_missing", mask_path)

    def __del__(self):
        """Clean up models when the object is deleted."""
        print("Unloading GroundedSAM models.")
        del self.sam_predictor
        del self.dino_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()