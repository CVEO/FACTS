# FACTS


## Installation

1.  Create the environment and install all dependencies:
    ```sh
    # First, create and activate a Python 3.12 virtual environment
    uv venv --python 3.12
    source .venv/bin/activate
    
    # After activating the environment, install all dependencies in one step
    # (Modify the PyTorch-related lines according to your system/CUDA version)
    uv pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        -r requirements.txt
    ```
    *Note: On Windows, the activation command is `.venv\Scripts\activate`.*

2.  Install GroundedSAM:
    ```sh
    .venv\Scripts\activate
    cd modules
    git clone https://github.com/SoluteToNight/GroundedSam4FACT.git
    cd GroundedSam4FACT
    uv pip install -e .
    uv pip install --no-build-isolation -e grounding_dino
    ```
    For more details, see the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repository.

## Usage

The main script for running the processing workflow is `demo.py`.

1.  Place your 3D model assets (e.g., `.obj`, `.mtl`, and texture images) into a subdirectory inside the `obj` folder. See `obj/example1` for reference.

2.  Run the script from the command line. You can specify the input directory for your model and where to save the output.

    ```sh
    python demo.py --input ./obj/example1 --output ./outputs/example1
    ```

### Command-Line Arguments

-   `--input`: Path to the directory containing the input model and textures.
-   `--output`: Directory where processed textures will be saved.
-   `--device`: Computation device, `cuda` or `cpu` (Default: `cuda`).
