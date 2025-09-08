import cv2
import numpy as np
from .node import BaseNode
from . import utils

def lab_brightness_normalization(img_list):
    """
    Normalizes brightness across multiple images using mean and standard deviation
    of the L-channel in the LAB color space. This method is suitable for images
    with similar lighting conditions that need fine-tuning.

    Processing Pipeline:
    1. Pre-computation: Gathers brightness statistics (mean, std dev) from all images.
    2. Adjustment: Applies normalization using global statistics.
    """
    print("Phase 1/2: Calculating global brightness statistics...")

    all_mu = []
    all_sigma = []

    for img in img_list:
        bgr_img = utils.pil_to_cv2_bgr(img.img_data)
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel = lab_img[:, :, 0]

        l_float = l_channel.astype(np.float64)
        mu = np.mean(l_float)
        sigma = np.std(l_float, ddof=1)

        all_mu.append(mu)
        all_sigma.append(sigma)

    if not all_mu:
        print("No valid images to process for illumination normalization.")
        return

    # Use median for robustness against outliers
    global_mu = np.median(all_mu)
    global_sigma = np.median(all_sigma)
    print(f"Global statistics: Target Mean (μ)={global_mu:.2f}, Target Std Dev (σ)={global_sigma:.2f}")

    print("\nPhase 2/2: Applying brightness normalization...")

    for i, img in enumerate(img_list):
        bgr_img = utils.pil_to_cv2_bgr(img.img_data)
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel = lab_img[:, :, 0].astype(np.float64)

        mu = all_mu[i]
        sigma = all_sigma[i]

        # Adjustment formula: L' = (σ_global/σ) * (L - μ) + μ_global
        if sigma < 1e-6:  # Avoid division by zero for uniform color images
            adjusted_l = l_channel
        else:
            adjusted_l = (global_sigma / sigma) * (l_channel - mu) + global_mu

        # Clip values to the valid 0-255 range and convert type
        adjusted_l = np.clip(adjusted_l, 0, 255).astype(np.uint8)

        # Merge channels and convert back to PIL image
        lab_img[:, :, 0] = adjusted_l
        result_bgr = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        img.img_data = utils.cv2_bgr_to_pil(result_bgr)
        print(f"  - Normalized {img.name}")

class Illumination(BaseNode):
    """
    A workflow node for advanced brightness normalization across a set of images.
    It uses the mean and standard deviation of the LAB L-channel to make the
    illumination consistent.
    """
    def __init__(self, image_processors):
        super().__init__(image_processors)

    def process(self):
        """
        Executes the brightness normalization process on the images.
        """
        if len(self.image_processors) < 2:
            print("Only one image provided, no illumination balancing needed.")
            return
            
        print("---'Running Advanced Illumination Normalization'---")
        lab_brightness_normalization(self.image_processors)
        print("---'Illumination Normalization Complete'---")