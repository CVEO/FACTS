import io
import json
import os
import random
import requests
import uuid
import websocket
from PIL import Image

from .node import BaseNode


# --- Network request functions ---
def upload_image(server_address, image_path, overwrite=True):
    """Uploads an image from a file path to the ComfyUI server."""
    filename = os.path.basename(image_path)
    with open(image_path, 'rb') as f:
        files = {'image': (filename, f, 'image/png'), 'overwrite': (None, str(overwrite).lower())}
        response = requests.post(f"http://{server_address}/upload/image", files=files)
        response.raise_for_status()
        return response.json()


def upload_image_from_object(server_address, image_object, filename, overwrite=True):
    """Uploads a PIL image object to the ComfyUI server."""
    byte_arr = io.BytesIO()
    image_object.save(byte_arr, format='PNG')
    byte_arr.seek(0)

    files = {'image': (filename, byte_arr, 'image/png'), 'overwrite': (None, str(overwrite).lower())}
    response = requests.post(f"http://{server_address}/upload/image", files=files)
    response.raise_for_status()
    return response.json()


def queue_prompt(server_address, prompt, client_id):
    """Queues a prompt on the ComfyUI server."""
    p = {"prompt": prompt, "client_id": client_id}
    response = requests.post(f"http://{server_address}/prompt", json=p)
    response.raise_for_status()
    return response.json()


def get_image(server_address, filename, subfolder, folder_type):
    """Retrieves an image from the ComfyUI server."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"http://{server_address}/view", params=data)
    response.raise_for_status()
    return response.content


def get_history(server_address, prompt_id):
    """Gets the execution history for a prompt from the ComfyUI server."""
    response = requests.get(f"http://{server_address}/history/{prompt_id}")
    response.raise_for_status()
    return response.json()


class Diffusion(BaseNode):
    """
    A workflow node that uses a remote ComfyUI server to process images based on a predefined workflow.
    """

    def __init__(self, image_processors, server_address="127.0.0.1:8188",
                 workflow_file="comfy-workflow/refine-workflow.json"):
        """
        Initializes the remote diffusion node.

        Args:
            image_processors: A list of image processor objects to work on.
            server_address (str): The address of the ComfyUI server.
            workflow_file (str): Path to the ComfyUI workflow JSON file.
        """
        super().__init__(image_processors)
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.workflow_template = self._load_workflow(workflow_file)

    def _load_workflow(self, workflow_file):
        """Loads the ComfyUI workflow from a JSON file."""
        print(f"Loading workflow from: {workflow_file}")
        try:
            # Assuming the workflow file is in the project root, which is the parent of the 'workflow' directory
            path = os.path.join(os.path.dirname(__file__), '..', workflow_file)
            if not os.path.exists(path):
                # Fallback to the provided path directly, in case it's absolute or relative to CWD
                path = workflow_file

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Workflow file '{workflow_file}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{workflow_file}'.")
            return None

    def process(self):
        """
        Processes each image using the remote ComfyUI server and the loaded workflow.
        """
        if not self.workflow_template:
            print("ComfyUI workflow template is not loaded. Skipping process.")
            return

        print("Applying remote diffusion (ComfyUI)...")

        for img_processor in self.image_processors:
            print(f"  - Processing {img_processor.name}...")

            ws = None
            try:
                image_info = upload_image_from_object(
                    self.server_address,
                    img_processor.img_data.convert("RGB"),
                    f"{self.client_id}_{img_processor.name}.png"
                )

                jsonwf = self.workflow_template.copy()
                jsonwf["2"]["inputs"]["image"] = image_info['name']

                # Hardcoded prompts from the original script
                jsonwf["15"]["inputs"][
                    "text"] = "Enhance the building facade by straightening and aligning all structural lines to follow perfect horizontal and vertical axes, ensuring no distortions, warping, or misalignment in the wall and window structures. Redraw and refine the windows, ensuring they are standard rectangular shapes with sharp, clean edges and clear alignment along the coordinate axes. Adjust window textures to include reflective glass or subtle blinds, preserving uniformity in size, style, and spacing across the facade. Transform dark, irregular areas into properly aligned and realistic windows that match the original style and dimensions.\nRefine the wall texture to be smooth, uniform, and free of irregularities, such as stains, cracks, or white patches, while preserving the building's natural color and material consistency. Remove any text, advertisements, or obstructive elements on the wall to create a clean and cohesive surface. Ensure air conditioning units and other exterior details remain intact, sharp, and integrated into the design. Eliminate all shadows to create even, natural lighting across the entire facade.\n\nSmooth and refine the building facade by eliminating all wrinkles, folds, or irregularities in the wall texture. Ensure the wall surface appears clean, uniform, and naturally smooth, while maintaining the original material’s texture and color consistency. Retain and subtly adjust the original windows to ensure they remain standard rectangular shapes with perfectly straight, clean edges. Preserve all other exterior elements, such as air conditioning units, ensuring they are sharp and well-integrated into the improved facade. Remove all shadows and distortions to create a uniformly lit, cohesive, and polished appearance.\nEnsure that the overall color tone of the facade remains consistent with the original, preserving the natural hues and materials used in the building's design. Maintain the same lighting and color scheme throughout t... [truncated]"
                jsonwf["16"]["inputs"][
                    "text"] = "blurry, distorted windows, low quality, curved lines, non-rectangular windows, heavy window grids or bars, unrealistic appearance, black spots on the wall, irregular shapes, stains, blotches, chaotic patterns, uneven color distribution, wall damage, or cracks"
                jsonwf["37"]["inputs"][
                    "text"] = "Ultra-precise architectural facade blueprint, (straightened lines:1.3), (perfect 90-degree angles:1.2), (mathematically aligned grid patterns), (crisp vector-style linework), (industrial-grade precision), (uniform window rectangles:1.1), (flawless horizontal/vertical alignment), (sharp edges:1.2), (CAD-quality drafting), (geometric perfection), (zero distortion), (symmetrical patterns), (clean technical drawing), (anodized aluminum texture), (glass panel reflections), (parametric design style)"
                jsonwf["38"]["inputs"][
                    "text"] = "curved lines, unrealistic appearance, irregular shapes, stains, blotches, chaotic patterns"
                jsonwf["53"]["inputs"][
                    "text"] = "Ultra-precise architectural facade blueprint, (straightened lines:1.3), (perfect 90-degree angles:1.2), (mathematically aligned grid patterns), (crisp vector-style linework), (industrial-grade precision), (uniform window rectangles:1.1), (flawless horizontal/vertical alignment), (sharp edges:1.2), (CAD-quality drafting), (geometric perfection), (zero distortion), (symmetrical patterns), (clean technical drawing), (anodized aluminum texture), (glass panel reflections), (parametric design style)"
                jsonwf["54"]["inputs"][
                    "text"] = "(crooked lines:1.3), (warped surfaces), (uneven edges), (hand-drawn sketch), (organic shapes), (blurry textures), (curved windows), (perspective distortion), (rustic style), (aged materials), (watercolor effect), (brush strokes), (deformed geometry), (asymmetrical elements), (freehand patterns)"
                jsonwf["73"]["inputs"][
                    "text"] = "Blurry, distorted windows, low quality, curved lines, non-rectangular windows, heavy window grids or bars, unrealistic appearance, black spots on the wall, irregular shapes, stains, blotches, chaotic patterns, uneven color distribution, wall damage, cracks, shadows, reflections, unnatural lighting, uneven lighting, distorted textures, visible text or advertisements, obstructive elements, 3D effects, depth effects, perspective distortions"
                jsonwf["74"]["inputs"][
                    "text"] = "Enhance the building facade by straightening and aligning all structural lines to follow perfect horizontal and vertical axes, ensuring no distortions, warping, or misalignment in the wall and window structures. Redraw and refine the windows, ensuring they are standard rectangular shapes with sharp, clean edges and clear alignment along the coordinate axes. Adjust window textures to include reflective glass or subtle blinds, preserving uniformity in size, style, and spacing across the facade. Transform dark, irregular areas into properly aligned and realistic windows that match the original style and dimensions.\nRefine the wall texture to be smooth, uniform, and free of irregularities, such as stains, cracks, or white patches, while preserving the building's natural color and material consistency. Remove any text, advertisements, or obstructive elements on the wall to create a clean and cohesive surface. Ensure air conditioning units and other exterior details remain intact, sharp, and integrated into the design. Eliminate all shadows to create even, natural lighting across the entire facade.\n\nSmooth and refine the building facade by eliminating all wrinkles, folds, or irregularities in the wall texture. Ensure the wall surface appears clean, uniform, and naturally smooth, while maintaining the original material’s texture and color consistency. Retain and subtly adjust the original windows to ensure they remain standard rectangular shapes with perfectly straight, clean edges. Preserve all other exterior elements, such as air conditioning units, ensuring they are sharp and well-integrated into the improved facade. Remove all shadows and distortions to create a uniformly lit, cohesive, and polished appearance.\nPreserve the Chinese characters on the wall exactly as they appear, ensuring they are not altered, removed, or distorted. Maintain the integrity of the text, keeping it sharp and clear, with no changes to the size, s... [truncated]"
                jsonwf["87"]["inputs"][
                    "text"] = "Blurry, distorted windows, low quality, curved lines, non-rectangular windows, heavy window grids or bars, unrealistic appearance, black spots on the wall, irregular shapes, stains, blotches, chaotic patterns, uneven color distribution, wall damage, cracks, shadows, reflections, unnatural lighting, uneven lighting, distorted textures, visible text or advertisements, obstructive elements, 3D effects, depth effects, a perspective distortions"
                jsonwf["88"]["inputs"][
                    "text"] = "Enhance the building facade by straightening and aligning all structural lines to follow perfect horizontal and vertical axes, ensuring no distortions, warping, or misalignment in the wall and window structures. Redraw and refine the windows, ensuring they are standard rectangular shapes with sharp, clean edges and clear alignment along the coordinate axes. Adjust window textures to include reflective glass or subtle blinds, preserving uniformity in size, style, and spacing across the facade. Transform dark, irregular areas into properly aligned and realistic windows that match the original style and dimensions.\nRefine the wall texture to be smooth, uniform, and free of irregularities, such as stains, cracks, or white patches, while preserving the building's natural color and material consistency. Remove any text, advertisements, or obstructive elements on the wall to create a clean and cohesive surface. Ensure air conditioning units and other exterior details remain intact, sharp, and integrated into the design. Eliminate all shadows to create even, natural lighting across the entire facade.\n\nSmooth and refine the building facade by eliminating all wrinkles, folds, or irregularities in the wall texture. Ensure the wall surface appears clean, uniform, and naturally smooth, while maintaining the original material’s texture and color consistency. Retain and subtly adjust the original windows to ensure they remain standard rectangular shapes with perfectly straight, clean edges. Preserve all other exterior elements, such as air conditioning units, ensuring they are sharp and well-integrated into the improved facade. Remove all shadows and distortions to create a uniformly lit, cohesive, and polished appearance.\nPreserve the Chinese characters on the wall exactly as they appear, ensuring they are not altered, removed, or distorted. Maintain the integrity of the text, keeping it sharp and clear, with no changes to the size, s... [truncated]"

                for seed_node in ["20", "43", "59", "75", "89"]:
                    if seed_node in jsonwf:
                        jsonwf[seed_node]["inputs"]["seed"] = random.randint(1, 99999999999)

                prompt_response = queue_prompt(self.server_address, jsonwf, self.client_id)
                prompt_id = prompt_response['prompt_id']

                ws = websocket.create_connection(f"ws://{self.server_address}/ws?clientId={self.client_id}", timeout=20)
                while True:
                    out = ws.recv()
                    if isinstance(out, str):
                        message = json.loads(out)
                        if message['type'] == 'executing' and message['data']['node'] is None and message['data'][
                            'prompt_id'] == prompt_id:
                            break

                history = get_history(self.server_address, prompt_id)[prompt_id]

                result_image = None
                for node_id in history['outputs']:
                    if node_id == "84":
                        node_output = history['outputs'][node_id]
                        if 'images' in node_output and node_output['images']:
                            image_spec = node_output['images'][0]
                            image_data = get_image(self.server_address, image_spec['filename'], image_spec['subfolder'],
                                                   image_spec['type'])
                            result_image = Image.open(io.BytesIO(image_data))
                            break

                if result_image:
                    img_processor.img_data = result_image
                    print(f"  - Successfully processed {img_processor.name}")
                else:
                    print(f"  - Processing finished but no output image found for {img_processor.name}")

            except Exception as e:
                print(f"  - An error occurred while processing {img_processor.name}: {e}")
            finally:
                if ws:
                    ws.close()

    def __del__(self):
        """Destructor."""
        pass
