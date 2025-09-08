import PIL
import numpy as np
import cv2 as cv
from PIL import Image
import os
import requests


class TextureImage:
    def __init__(self, path: str):
        self.img_path: str = path
        self.img_data: Image.Image = None
        self.tmp_data = None # Can store intermediate data, e.g., numpy arrays
        self.brightness = None
        self.name = os.path.basename(path)
        self.building_obj = None
        self.load_image()
        print(self)

    def __str__(self):
        return f"TextureImage loaded from: {self.img_path}"

    def load_image(self):
        """Loads an image from a local path or a URL."""
        if isinstance(self.img_path, str):
            try:
                if self.img_path.startswith("http://") or self.img_path.startswith("https://"):
                    response = requests.get(self.img_path, stream=True)
                    response.raise_for_status()
                    self.img_data = Image.open(response.raw).convert("RGB")
                elif os.path.isfile(self.img_path):
                    self.img_data = Image.open(self.img_path).convert("RGB")
                else:
                    raise FileNotFoundError(f"Image file not found at path: {self.img_path}")
            except Exception as e:
                print(f"Error loading image {self.img_path}: {e}")
                # Create a placeholder black image on failure
                self.img_data = Image.new('RGB', (256, 256), color = 'black')
                self.name = f"error_{self.name}"

    def update_image(self, new_image: Image.Image):
        """Safely updates the main image data with a new PIL Image."""
        if not isinstance(new_image, Image.Image):
            raise TypeError("Updated image must be a PIL.Image instance.")
        self.img_data = new_image
        self.tmp_data = None

    def save(self, output_dir: str):
        """Saves the current image data to the specified directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, self.name)
        self.img_data.save(save_path)
        print(f"Saved image to {save_path}")

    def tmp_save(self, data: np.ndarray = None, suffix: str = "tmp"):
        """Saves a temporary image (from numpy array) for debugging."""
        tmp_path = os.path.join(self.building_obj.temp_path, f"{os.path.splitext(self.name)[0]}_{suffix}.png")
        image_to_save = None
        if data is None:
            # If no data is provided, save the current state of img_data
            image_to_save = self.img_data
        elif isinstance(data, np.ndarray):
            # Assuming data is a BGR numpy array from OpenCV, convert to RGB for saving
            if data.ndim == 2: # Grayscale image
                image_to_save = Image.fromarray(data, mode='L')
            else:
                image_to_save = Image.fromarray(cv.cvtColor(data, cv.COLOR_BGR2RGB))
        
        if image_to_save:
            image_to_save.save(tmp_path)

    # The following methods are for compatibility with potential legacy use,
    # but direct use of utils is preferred for clarity.
    def convert_to_cv2_bgr(self) -> np.ndarray:
        """Converts the PIL image to an OpenCV-compatible BGR NumPy array."""
        return np.array(self.img_data.convert("RGB"))[:, :, ::-1]

    def update_from_cv2_bgr(self, bgr_array: np.ndarray):
        """Updates the image from an OpenCV-compatible BGR NumPy array."""
        rgb_array = cv.cvtColor(bgr_array, cv.COLOR_BGR2RGB)
        self.img_data = Image.fromarray(rgb_array)
