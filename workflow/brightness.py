import numpy as np
from PIL import Image

from .node import BaseNode
from . import utils
import cv2 as cv
class Brightness(BaseNode):
    """
    A workflow node to analyze and normalize the brightness of a list of images
    using the L-channel of the LAB color space.
    """

    def __init__(self, image_processors):
        super().__init__(image_processors)
        self.brightness_stats = {}

    def _calculate_lab_brightness(self, lab_image: np.ndarray) -> float:
        """Calculates the mean brightness from the L-channel of a LAB image."""
        l_channel = lab_image[:, :, 0]
        return np.mean(l_channel)

    def _balance_lab_brightness(self, lab_image: np.ndarray, value: float) -> np.ndarray:
        """Adjusts the L-channel of a LAB image by a given value."""
        l_channel, a_channel, b_channel = cv.split(lab_image)
        
        # Perform calculations in float32 to prevent overflow/underflow
        l_float = l_channel.astype(np.float32)
        l_float += value
        
        # Clip the values to the valid 0-255 range for uint8
        l_adjusted = np.clip(l_float, 0, 255).astype(np.uint8)
        
        # Merge the adjusted L-channel with the original A and B channels
        return cv.merge([l_adjusted, a_channel, b_channel])

    def process(self):
        """Main processing loop for LAB brightness normalization."""
        print("Analyzing image brightness using LAB color space...")
        
        # 1. Convert to LAB and calculate brightness for all images
        for img_processor in self.image_processors:
            bgr_img = utils.pil_to_cv2_bgr(img_processor.img_data)
            lab_img = utils.bgr_to_lab(bgr_img)
            brightness = self._calculate_lab_brightness(lab_img)
            self.brightness_stats[img_processor.name] = {'brightness': brightness, 'lab': lab_img}
            print(f"  - {img_processor.name}: Brightness (L*) = {brightness:.2f}")

        if len(self.image_processors) < 2:
            print("Only one image provided, no balancing needed.")
            return

        # 2. Find average brightness and identify images that need correction
        all_brightness_values = [stats['brightness'] for stats in self.brightness_stats.values()]
        average_brightness = np.mean(all_brightness_values)
        print(f"Average brightness across all images: {average_brightness:.2f}")

        # 3. Adjust and update images
        print("Balancing brightness...")
        for img_processor in self.image_processors:
            stats = self.brightness_stats[img_processor.name]
            brightness_diff = average_brightness - stats['brightness']

            # Only adjust if the difference is significant (e.g., > 5 L* units)
            if abs(brightness_diff) > 5:
                print(f"  - Adjusting {img_processor.name} by {brightness_diff:.2f} L* units")
                adjusted_lab = self._balance_lab_brightness(stats['lab'], brightness_diff)
                
                # Convert back to BGR and then to PIL to update the ImageProcessor
                adjusted_bgr = utils.lab_to_bgr(adjusted_lab)
                img_processor.img_data = utils.cv2_bgr_to_pil(adjusted_bgr)
            else:
                print(f"  - {img_processor.name} is already balanced.")