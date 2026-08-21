"""
Example algorithm for Grand Challenge.

Edit this file to implement your algorithm. It contains:

  init_model() — Called once at server startup. Load your model here.
  run(model)   — Called each time the /invoke endpoint is hit. This should read input
                 from /input, run inference, and write output to /output.

Your algorithm's interfaces:

  interf0:
    Inputs:

      - /input/images/t1-brain-mri

      - /input/stroke-metadata.json

    Outputs:

      - /output/images/stroke-lesion-segmentation
      - /output/images/lesion-probability-map


Which interface is active for a given invocation is determined by the
inputs.json file at /input/inputs.json — a system-generated file that
describes the input sockets provided for this case.

To test locally:  ./do_test_run.sh
To save for upload to GC:   ./do_save.sh

Any implementation will do as long as it produces the output as prescribed by the interface.

For more details see:
  https://grand-challenge.org/documentation/algorithms/
  https://grand-challenge.org/documentation/runtime-environment/
"""

import glob
import json
from pathlib import Path

import numpy
import SimpleITK
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

M2_MODEL = "nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres"
M3_MODEL = "nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres"
M4_MODEL = "nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres"
ENSEMBLE_WEIGHTS = {
    M2_MODEL: 0.20,
    M3_MODEL: 0.30,
    M4_MODEL: 0.50,
}


# def _show_torch_cuda_info():
#     print("=+=" * 10)
#     print("Collecting Torch CUDA information")
#     print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
#     if available:
#         print(f"\tnumber of devices: {torch.cuda.device_count()}")
#         print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
#         print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
#     print("=+=" * 10)


# def init_model():
#     """Load and return your model.
#
#     This is called once by app.py during server startup (before /health returns 200).
#     The model is then reused across all /invoke calls — so load it here,
#     not inside the run() function.
#     """
#     _show_torch_cuda_info()
#
#     # Example how to set torch to use the GPU (if available)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     model = torch.nn.Linear(10, 1).to(device)
#
#     # Your model will be extracted to `model_dir` at runtime on Grand Challenge.
#     # When testing locally, the local `./model` directory is mounted here.
#     # When you're ready for testing on Grand Challenge, upload your model as a tarball
#     # via Your algorithm > Models on Grand Challenge.
#     # For now, just verify that we can read from the model directory
#     model_dir = Path("/opt/ml/model")
#     with open(
#         model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
#     ) as f:
#         print(f.read())
#
#     return model

def init_model():
    """Load and return your model.

    This is called once by app.py during server startup (before /health returns 200).
    The model is then reused across all /invoke calls — so load it here,
    not inside the run() function.
    """

    model_folders = sorted({p.parent for p in MODEL_PATH.rglob("dataset.json")})
    model_folders = [
        p for p in model_folders
        if (p / "plans.json").is_file()
        and any(p.glob("fold_*/checkpoint_best.pth"))
    ]
    model_folders = {p.name: p for p in model_folders}
    if set(model_folders) != set(ENSEMBLE_WEIGHTS):
        raise RuntimeError(
            "Expected the M2, M3 and M4 nnU-Net models below /opt/ml/model, "
            f"found: {sorted(model_folders)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictors = {}
    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        model_folder = model_folders[model_name]
        print(f"Loading {model_name} (weight={weight:.2f}) from {model_folder} on {device}")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=device.type == "cuda",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        use_folds = ("all",) if (model_folder / "fold_all").is_dir() else None
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=use_folds,
            checkpoint_name="checkpoint_best.pth",
        )
        predictors[model_name] = predictor
    return predictors


def run(model):
    """This is called each time the /invoke endpoint is called.
    This should read input from /input, run inference, and write output to /output.
    """
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler(model)


def interf0_handler(model):
    # Read the input and get the original SimpleITK image reference to preserve affine and header
    t1_image, t1_data = load_image_file_as_array_and_image(
        location=INPUT_PATH / "images/t1-brain-mri",
    )

    # Load and log input stroke metadata
    input_stroke_metadata = load_json_file(
        location=INPUT_PATH / "stroke-metadata.json",
    )

    # =========================================================================
    # METADATA NOTES:
    # 1) Unlike the training dataset where the acquisition site key is referred
    #    to as 'SITE', in Grand Challenge inputs it is named 'CENTER'.
    # 2) Note that metadata values (e.g., DAYS_POST_STROKE, CHRONICITY) may
    #    contain `null` (None in Python) values that your pipeline must handle.
    # =========================================================================
    print("=+=" * 10)
    print("Loaded Stroke Metadata:")
    print(json.dumps(input_stroke_metadata, indent=2))
    print("=+=" * 10)

    # nnU-Net expects channel-first input and spacing in NumPy axis order (z, y, x).
    input_array = t1_data[numpy.newaxis].astype(numpy.float32, copy=False)
    image_properties = {"spacing": list(t1_image.GetSpacing())[::-1]}
    probabilities = None
    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        _, model_probabilities = model[model_name].predict_single_npy_array(
            input_image=input_array,
            image_properties=image_properties,
            save_or_return_probabilities=True,
        )
        weighted_probabilities = model_probabilities.astype(
            numpy.float32, copy=False
        ) * weight
        if probabilities is None:
            probabilities = weighted_probabilities
        else:
            probabilities += weighted_probabilities

    binary_segmentation_mask = numpy.argmax(probabilities, axis=0)
    binary_segmentation_mask = binary_segmentation_mask.astype(numpy.uint8, copy=False)
    probability_map_data = probabilities[1].astype(numpy.float32, copy=False)

    # Save outputs keeping the exact same spatial metadata (affine/header) as the input t1 image
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=binary_segmentation_mask,
        reference_image=t1_image,
    )

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/lesion-probability-map",
        array=probability_map_data,
        reference_image=t1_image,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array_and_image(*, location):
    # Use SimpleITK to read a file and return both the image object and array
    input_files = (
        glob.glob(str(location / "*.mha"))
        + glob.glob(str(location / "*.nii.gz"))
        + glob.glob(str(location / "*.nii"))
    )
    if not input_files:
        raise FileNotFoundError(f"No valid image file found in {location}")

    image = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array and return along with the SimpleITK image
    return image, SimpleITK.GetArrayFromImage(image)


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    
    # Preserve original spacing, direction, and origin (affine/header) from input if provided
    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())

    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
