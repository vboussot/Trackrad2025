"""
Edit this file to implement your algorithm. 

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
"""
from pathlib import Path
import numpy as np
import torch
import os
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

RESOURCE_PATH = Path("resources")  # load weights and other resources from this directory

def prepare_video(array: np.ndarray):
    video = []
    for frame in array:
        norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-5)
        frame_uint8 = (norm * 255).astype(np.uint8)  # [H, W]
        rgb = np.stack([frame_uint8] * 3, axis=0)    # [3, H, W]
        video.append(torch.from_numpy(rgb))
    return torch.stack(video)  # [T, 3, H, W]

def save_images(video_array, output_folder):
    for i, frame in enumerate(video_array):
        image = Image.fromarray(frame.transpose((1,2,0)))
        image.save(os.path.join(output_folder, f'{i:04d}.jpg'))

def load_sam2():
    checkpoint = "./SAM2.1_b+_finetune.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device = torch.device("cuda"))
    return predictor

def run_algorithm(frames: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Implement your algorithm here.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
    - target (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
    """
    predictor = load_sam2()

    frames = frames.transpose((2,0,1))
    video_tensor = prepare_video(frames)
    tmp_dir = "./tmp/"
    os.makedirs(tmp_dir, exist_ok=True)
    save_images(video_tensor.numpy(), tmp_dir)
    
    mask_np =target.squeeze()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(tmp_dir)

        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            mask=mask_np)
                
        masks_out = []
        for _, _, out_mask_logits in predictor.propagate_in_video(state):
            masks_out.append((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8))  # [H, W]
    return np.stack(masks_out).squeeze().transpose((1,2,0))  # [T, H, W]
