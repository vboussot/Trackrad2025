"""
The following is an entrypoint script for an algorithm.

It load the input data, runs the algorithm and saves the output data.

The actual algorithm is implemented in the model.py file.

You should not need to modify this file.

"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import time

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

def run():
    loading_start_time = time.perf_counter()

    # Read the inputs
    input_frame_rate = load_json_file(
         location=INPUT_PATH / "frame-rate.json",
    )
    input_magnetic_field_strength = load_json_file(
         location=INPUT_PATH / "b-field-strength.json",
    )
    input_scanned_region = load_json_file(
         location=INPUT_PATH / "scanned-region.json",
    )
    input_mri_linac_series = load_image_file_as_array(
        location=INPUT_PATH / "images/mri-linacs",
    )

    input_mri_linac_target = load_image_file_as_array(
        location=INPUT_PATH / "images/mri-linac-target",
    )

    print(f"Runtime loading:   {time.perf_counter() - loading_start_time:.5f} s")

    from model import run_algorithm

    algo_start_time = time.perf_counter()

    output_mri_linac_series_targets = run_algorithm(frames=input_mri_linac_series, 
                                                    target=input_mri_linac_target,
                                                    frame_rate=input_frame_rate,
                                                    magnetic_field_strength=input_magnetic_field_strength,
                                                    scanned_region=input_scanned_region)

    print(f"Runtime algorithm: {time.perf_counter() - algo_start_time:.5f} s")

    writing_start_time = time.perf_counter()

    # Save the output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/mri-linac-series-targets",
        array=output_mri_linac_series_targets,
    )
    print(f"Runtime writing:   {time.perf_counter() - writing_start_time:.5f} s")
    
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, 'r') as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())