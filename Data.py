from huggingface_hub import snapshot_download
from konfai.utils.dataset import Dataset
import os
import SimpleITK as sitk
import shutil

if __name__ == "__main__":
    snapshot_download(
        repo_id="LMUK-RADONC-PHYS-RES/TrackRAD2025",
        repo_type="dataset",
        allow_patterns="trackrad2025_labeled_training_data/**",
        local_dir="./trackrad2025_labeled_training_data"
    )

    dataset = Dataset("./Dataset/", "mha")

    names = os.listdir("./trackrad2025_labeled_training_data/trackrad2025_labeled_training_data/")

    for name in names:
        image = sitk.ReadImage(f"./trackrad2025_labeled_training_data/trackrad2025_labeled_training_data/{name}/images/{name}_frames.mha")
        labels = sitk.ReadImage(f"./trackrad2025_labeled_training_data/trackrad2025_labeled_training_data/{name}/targets/{name}_labels.mha")
        mask = sitk.ReadImage(f"./trackrad2025_labeled_training_data/trackrad2025_labeled_training_data/{name}/targets/{name}_first_label.mha")

        dataset.write("IMAGE", name, image)
        dataset.write("LABELS", name, labels)
        dataset.write("MASK", name, mask)

    shutil.rmtree("./trackrad2025_labeled_training_data/")