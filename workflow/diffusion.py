import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

from .node import BaseNode

class Diffusion(BaseNode):
    """
    A workflow node that applies a Stable Diffusion Img2Img model to a list of images.
    """
    def __init__(self, image_processors, model_path="models/architecturerealmix"):
        super().__init__(image_processors)
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the Stable Diffusion pipeline model."""
        print(f"Loading Diffusion model from: {self.model_path}")
        try:
            self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            self.model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)
            print("Diffusion model loaded successfully.")
        except Exception as e:
            print(f"Error loading Diffusion model: {e}")
            self.model = None

    def process(self, prompt="high quality image", negative_prompt="blur, distortion, shadow", strength=0.2, steps=45):
        """
        Processes each image using the Img2Img pipeline.

        Args:
            prompt (str): The positive prompt for the diffusion model.
            negative_prompt (str): The negative prompt.
            strength (float): How much to transform the initial image (0.0 to 1.0).
            steps (int): The number of inference steps.
        """
        if not self.model:
            print("Diffusion model is not loaded. Skipping process.")
            return

        print("Applying Stable Diffusion Img2Img...")
        for img_processor in self.image_processors:
            print(f"  - Processing {img_processor.name}...")
            
            # Ensure the image is in RGB format, as expected by the model
            initial_image = img_processor.img_data.convert("RGB")

            # Run the model
            result_image = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=initial_image,
                strength=strength,
                num_inference_steps=steps
            ).images[0]

            # Update the image processor with the new image
            img_processor.img_data = result_image

    def __del__(self):
        """Clean up the model when the object is deleted."""
        if hasattr(self, 'model') and self.model is not None:
            print("Unloading Diffusion model.")
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()