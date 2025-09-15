import os
from workflow import *
import torch
from argparse import ArgumentParser, Namespace
from PIL import Image
from img_class import TextureImage as timg
from DataLoader import load_data
import sys

from workflow.brightness import Brightness
from workflow.diffusion import Diffusion
from workflow.illumination import Illumination
from workflow.segment import Segment

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 将 models 目录添加到 sys.path

def arg_parser() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default=r"E:\FACTS\obj\example1", help="input image dir")
    parser.add_argument("-o", "--output", default="outputs", help="output image dir")
    parser.add_argument("--device", default="cuda", help="device")
    return parser.parse_args()


def check_device(device):
    if device == "cpu":
        return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("No CUDA device is available, using CPU instead")
            return "cpu"
        return "cuda"


def main() -> None:
    args = arg_parser()
    script_path = os.getcwd()
    input_path = args.input
    output_path = args.output
    device = check_device(args.device)
    print("device:", device)

    img_list = []
    building = load_data(input_path, output_path)
    while True:
        try:
            bd = next(building)
            # bd.load_texture()
            img_list = bd.texture_list
            # The following is a configurable workflow.
            # You can uncomment or rearrange the nodes as needed.
            print("--- Starting Workflow ---")

            # Step 1: Segment the image to identify different parts (e.g., window, wall).
            # This process generates masks in the temp directory for potential later use.
            print("Step 1: Segmenting images...")
            Segment(img_list).process()

            print("Step 2: Normalizing brightness...")
            Brightness(img_list).process()

            print("Step 2a: Applying advanced illumination normalization...")
            # You can choose different methods: 'adaptive' (default), 'iterative', 'statistical'
            Illumination(img_list, method='adaptive').process()

            #  Step 3: Apply a diffusion model to enhance or alter textures.

            print("Step 3: Applying diffusion model...")
            Diffusion(img_list).process()

            print("--- Workflow Finished ---")
            for img in img_list:
                img.save(f"{bd.output_path}")
            # cal_grad.eval_by_grad(bd)
        except StopIteration:
            print("No more data")
            break


if __name__ == "__main__":
    main()
