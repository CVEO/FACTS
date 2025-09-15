import os
from os.path import split, join
from collections import Counter
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from diffusers import StableDiffusionInpaintPipeline
from dotenv import load_dotenv

from .node import BaseNode
from modules.GroundedSam4FACT.predict import load_sam2, load_dino, predict_mask

# Load environment variables from .env file
load_dotenv()

# Constants
upper_threshold = 0.9
lower_threshold = 0.1

def mask_rate(mask):
    """Calculates the ratio of non-zero pixels in a mask."""
    if mask.size == 0:
        return 0.0
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0
    ones = np.count_nonzero(mask)
    rate = ones / total_pixels
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
    print(f"Saved mask to {save_path}")

def auto_resize(image_processor, max_pixels=2500000):
    """Resizes an image if it exceeds a maximum pixel count, returns the new path and a resize flag."""
    img = image_processor.img_data
    if img.size[0] * img.size[1] <= max_pixels:
        return image_processor.img_path, False

    print(f"Resizing {image_processor.name} as it exceeds {max_pixels} pixels.")
    resize_scale = np.sqrt(max_pixels / (img.size[0] * img.size[1]))
    new_size = (int(img.size[0] * resize_scale), int(img.size[1] * resize_scale))
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    temp_dir = Path(image_processor.building_obj.temp_path)
    temp_dir.mkdir(exist_ok=True)
    resized_path = temp_dir / image_processor.name
    resized_img.save(resized_path, "PNG")
    return str(resized_path), True

def infer_missing(masks):
    """Performs morphological operations to find missing parts of a mask."""
    kernel = np.ones((3, 3), np.uint8)
    infered_missing_mask_uint8 = (masks > 0).astype('uint8') * 255
    opened_mask = cv2.morphologyEx(infered_missing_mask_uint8, cv2.MORPH_OPEN, kernel)
    infered_missing_bool = np.logical_not(opened_mask > 0)
    return infered_missing_bool

def analyse_mask(history, *arg):
    """Analyzes a history of mask predictions to find the optimal one by examining volatile and smooth intervals."""
    if len(history) < 2:
        print("Analyse_mask: Not enough data points (< 2) to analyze.")
        return (history[0]['mask'], None) if history and 0.01 < history[0]['rate'] < 0.80 else (None, None)

    valid_history = [his for his in history if 'threshold' in his and 'rate' in his and 'mask' in his]
    thresholds = np.array([his['threshold'] for his in valid_history])
    rates = np.array([his['rate'] for his in valid_history])

    sort_indices = np.argsort(thresholds)
    thresholds, rates = thresholds[sort_indices], rates[sort_indices]
    sorted_history = [valid_history[i] for i in sort_indices]

    probable_mask, probable_missing = [], []
    returned_masks, infered_missing_mask = None, None

    threshold_diffs = np.diff(thresholds)
    rate_diffs = np.diff(rates)
    change_rates = np.divide(rate_diffs, threshold_diffs, out=np.zeros_like(rate_diffs, dtype=float), where=threshold_diffs != 0)
    
    threshold_change_rate = 0.1
    volatile_intervals, smooth_intervals = [], []
    start_smooth_threshold = None

    for i in range(len(change_rates)):
        is_volatile = abs(change_rates[i]) > threshold_change_rate
        if is_volatile:
            volatile_intervals.append((thresholds[i], thresholds[i+1]))
            if start_smooth_threshold is not None:
                smooth_intervals.append((start_smooth_threshold, thresholds[i]))
                start_smooth_threshold = None
        else:
            if start_smooth_threshold is None:
                start_smooth_threshold = thresholds[i]
    if start_smooth_threshold is not None:
        smooth_intervals.append((start_smooth_threshold, thresholds[-1]))

    print(f"Volatile intervals: {volatile_intervals}")
    print(f"Smooth intervals: {smooth_intervals}")

    for start_th, end_th in smooth_intervals:
        indices = np.where((thresholds >= start_th) & (thresholds <= end_th))
        if not indices[0].any(): continue
        
        middle_iter = indices[0][len(indices[0]) // 2]
        middle_rate = sorted_history[middle_iter]['rate']
        masks = sorted_history[middle_iter]['mask']

        if 0.80 <= middle_rate < 0.99:
            probable_missing.append(infer_missing(masks))
        elif 0.05 < middle_rate < 0.80:
            probable_mask.append(masks)

    if probable_mask:
        returned_masks = probable_mask[len(probable_mask) // 2]
    if probable_missing:
        infered_missing_mask = probable_missing[0]

    for start_th, end_th in volatile_intervals:
        start_indices = np.where(thresholds == start_th)[0]
        if not start_indices.any(): continue
        start_iter = start_indices[0]
        
        start_rate = sorted_history[start_iter]['rate']
        end_rate = rates[np.where(thresholds == end_th)[0][0]]

        if (end_rate - start_rate) > 0.7 and arg:
            print(f"Refining volatile interval ({start_th:.2f}-{end_th:.2f})")
            # This part can be expanded with recursive refinement if needed
            returned_masks = sorted_history[start_iter]['mask']
        else:
            returned_masks = sorted_history[start_iter]['mask']

    return returned_masks, infered_missing_mask


class Segment(BaseNode):
    """A self-contained node for segmenting and inpainting images."""
    def __init__(self, image_processors, sd_strength=0.8, inpainting_model_path=None):
        super().__init__(image_processors)
        self.strength = sd_strength
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading GroundedSAM models for Segment node...")
        self.sam_predictor = load_sam2()
        self.dino_predictor = load_dino()
        print("GroundedSAM models loaded.")

        self.inpainting_pipe = None
        
        # Determine the final model path
        final_model_path = inpainting_model_path # Priority 1: Direct argument

        # Priority 2: .env variables, if no direct argument
        if not final_model_path:
            comfyui_path = os.getenv("PATH_TO_COMFYUI")
            model_path_from_env = os.getenv("INPAINTING_MODEL_PATH")
            
            if comfyui_path and model_path_from_env:
                # Join base path and relative model path
                final_model_path = join(comfyui_path, model_path_from_env.lstrip('/\\'))
            elif model_path_from_env: # Handle case where the env var is an absolute path
                final_model_path = model_path_from_env

        # Priority 3: Hardcoded default, if no path found yet
        if not final_model_path:
            final_model_path = r"E:/Download/architectureUrbanSdlife_v60.safetensors"
            print(f"Warning: No valid model path in args or .env. Using default: {final_model_path}")

        # Load the model using the determined path
        try:
            if final_model_path and os.path.exists(final_model_path):
                print(f"Attempting to load inpainting model from: {final_model_path}")
                self.inpainting_pipe = StableDiffusionInpaintPipeline.from_single_file(
                    final_model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                ).to(self.device)
                print("Inpainting model loaded successfully.")
                self.inpainting_pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled.")
            else:
                print(f"Error: Inpainting model file not found at {final_model_path}.")
        except Exception as e:
            print(f"Error loading inpainting model or enabling xformers: {e}")
            self.inpainting_pipe = None

    def process(self, prompt_list=["window", "missing", "black occlusion", "tree", "leaves"]):
        for img_processor in self.image_processors:
            print(f"--- Processing {img_processor.name} ---")
            
            img_path_for_prediction, resized = auto_resize(img_processor)
            original_image_pil = img_processor.img_data # Already a PIL image
            
            temp_path = img_processor.building_obj.temp_path
            Path(temp_path).mkdir(exist_ok=True)
            mask_path_base = os.path.join(temp_path, os.path.splitext(img_processor.name)[0])

            for prompt_seg in prompt_list:
                print(f"\nSegmenting for prompt: '{prompt_seg}'")
                predict_history = []
                box_threshold = lower_threshold
                
                while box_threshold <= upper_threshold:
                    try:
                        masks_bool = predict_mask(
                            prompt_seg, self.sam_predictor, self.dino_predictor, 
                            img_path_for_prediction, box_threshold, box_threshold
                        )
                        rate = mask_rate(masks_bool) if masks_bool is not None else 0.0
                        predict_history.append({'mask': masks_bool, 'rate': rate, 'threshold': box_threshold})
                        if (rate == 0.0 and box_threshold > lower_threshold + 0.05) or rate > 0.98:
                            break
                    except Exception as e:
                        print(f"Error during prediction at threshold {box_threshold:.2f}: {e}")
                        break
                    box_threshold += 0.05

                print(f"Analyzing {len(predict_history)} predictions...")
                analysis_args = (prompt_seg, self.sam_predictor, self.dino_predictor, img_path_for_prediction)
                final_mask_bool, missing_bool = analyse_mask(predict_history, *analysis_args)

                # --- Save masks before attempting inpainting --- 
                if final_mask_bool is not None:
                    save_masks_as_files(final_mask_bool, f"{prompt_seg}_final_mask", mask_path_base)
                if missing_bool is not None:
                    save_masks_as_files(missing_bool, f"{prompt_seg}_missing_mask", mask_path_base)

                # --- Determine target for inpainting --- 
                target_mask_bool, inpainting_prompt, negative_prompt = None, "", ""
                mask_suffix, output_suffix = "", ""

                if prompt_seg == "window" and final_mask_bool is not None:
                    target_mask_bool = final_mask_bool
                    inpainting_prompt = "photorealistic window, high detail"
                    mask_suffix, output_suffix = "window_mask", "window_inpainted"
                elif prompt_seg == "missing" and missing_bool is not None:
                    target_mask_bool = missing_bool
                    inpainting_prompt = "restore missing part of the building facade, consitent to the surrounding wall texture, Seamless continuation of the surrounding building facade texture, wall surface"
                    mask_suffix, output_suffix = "missing_mask", "missing_inpainted"
                elif prompt_seg == "black occlusion" and missing_bool is not None:
                    target_mask_bool = missing_bool
                    inpainting_prompt = "Remove the black occlusion. Fill the area seamlessly with the surrounding building facade wall texture, matching the existing wall surface."
                    negative_prompt = "black, occlusion, missing, billboard, advertisement, sign, text, logo, poster, graphic, letters, words, visual clutter, artifact, distortion"
                    mask_suffix, output_suffix = "occlusion_mask", "occlusion_removed_inpainted"
                elif prompt_seg == "tree" and missing_bool is not None:
                    target_mask_bool = missing_bool
                    inpainting_prompt = "Remove the tree part. Fill the area seamlessly with the surrounding building facade wall texture, matching the existing wall surface, ensure consistency with building facade."
                    negative_prompt = "black, occlusion, missing, billboard, advertisement, sign, text, logo, poster, graphic, letters, words, visual clutter, artifact, distortion"
                    mask_suffix, output_suffix = "tree_mask", "tree_removed_inpainted"
                elif prompt_seg == "leaves" and final_mask_bool is not None:
                    target_mask_bool = final_mask_bool
                    inpainting_prompt = "remove the green and colorful leaves, seamlessly blend with the surrounding building facade and ground, Seamless continuation of the surrounding building facade texture, plain wall surface"
                    mask_suffix, output_suffix = "leaves_mask", "leaves_inpainted"
                
                # --- Perform Inpainting (if possible) --- 
                if target_mask_bool is not None:
                    if self.inpainting_pipe is None:
                        print(f"Inpainting model not loaded. Skipping inpainting for '{prompt_seg}'.")
                        continue # Skip to the next prompt

                    print(f"Performing inpainting for: {prompt_seg}")
                    mask_np_uint8 = target_mask_bool.astype(np.uint8) * 255
                    
                    dilation_kernel = np.ones((9, 9), np.uint8)
                    dilated_mask_np_uint8 = cv2.dilate(mask_np_uint8, dilation_kernel, iterations=1)

                    mask_pil = Image.fromarray(dilated_mask_np_uint8).convert("L")
                    save_masks_as_files(dilated_mask_np_uint8, f"{mask_suffix}_dilated_final_for_inpainting", mask_path_base)

                    original_width, original_height = original_image_pil.size
                    pipe_width, pipe_height = (original_width // 8) * 8, (original_height // 8) * 8
                    
                    if pipe_width == 0 or pipe_height == 0:
                        print("Skipping inpainting due to zero dimensions after rounding.")
                        continue

                    image_for_pipe = original_image_pil.resize((pipe_width, pipe_height), Image.Resampling.LANCZOS)
                    mask_for_pipe = mask_pil.resize((pipe_width, pipe_height), Image.Resampling.NEAREST)

                    try:
                        pipe_args = {
                            "prompt": inpainting_prompt, "image": image_for_pipe, "mask_image": mask_for_pipe,
                            "strength": self.strength, "height": pipe_height, "width": pipe_width,
                            "num_inference_steps": 20
                        }
                        if negative_prompt:
                            pipe_args["negative_prompt"] = negative_prompt

                        result = self.inpainting_pipe(**pipe_args).images[0]
                        inpainted_image = result.resize((original_width, original_height), Image.Resampling.LANCZOS)
                        
                        inpainted_image_path = f"{mask_path_base}_{output_suffix}.png"
                        inpainted_image.save(inpainted_image_path)
                        print(f"Saved inpainted image to {inpainted_image_path}")

                    except Exception as e:
                        print(f"Error during inpainting pipeline execution: {e}")
                else:
                    print(f"No valid mask found for '{prompt_seg}', skipping inpainting.")
        
        print("\nProcessing finished for all images.")

    def __del__(self):
        """Clean up models when the object is deleted."""
        print("Unloading models.")
        del self.sam_predictor
        del self.dino_predictor
        if hasattr(self, 'inpainting_pipe'):
            del self.inpainting_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()