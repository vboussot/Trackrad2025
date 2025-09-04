[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-TrackRAD2025-yellow)](https://huggingface.co/VBoussot/Trackrad2025)

# TrackRAD2025: Real-time Tumor Tracking (6th place 🏅)

This repository provides the full code and configurations used for our submission to the **TrackRAD2025 Challenge**, focused on **real-time tumor tracking** in cine-MRI sequences.
Our approach is based on **SAM 2.1 (Segment Anything Model)**, adapted for the medical domain using **prompt-based fine-tuning** on the small labeled subset provided by the challenge.

---

## 📊 Results

### ✅ Local Validation Results

We evaluated several foundation model variants and prompting strategies on the annotated validation subset of TrackRAD2025.

#### 📌 Comparison between segmentation models:

| Metric         | Baseline | SAM 2.1 t | SAM 2.1 b+ | SAM 2.1 l | MedSAM2 | MedSAM2 liver | 
|----------------|----------|-----------|------------|-----------|---------|----------------|
| **DSC**        | 0.770    | 0.911     | 0.915      | **0.916** | 0.869   | 0.842          |
| **HD (mm)**    | 8.27     | 3.45      | 3.41       | **3.26**  | 19.76   | 29.62          |
| **ASD (mm)**   | 3.88     | 1.27      | 1.29       | **1.20**  | 17.01   | 27.32          |
| **CD (mm)**    | 6.42     | 1.51      | 1.46       | **1.34**  | 17.28   | 27.51          |

#### 📌 Effect of fine-tuning (on SAM 2.1 l):

| Metric         | SAM 2.1 l | Ensemble prompt | Fine-tuned SAM 2.1 l |
|----------------|-----------|------------------|------------------------|
| **DSC**        | 0.916     | 0.908            | **0.929**              |
| **HD (mm)**    | 3.26      | 3.51             | **2.78**               |
| **ASD (mm)**   | 1.20      | 1.29             | **1.00**               |
| **CD (mm)**    | 1.34      | 1.39             | **1.16**               |

Fine-tuning the SAM 2.1 l model improved all metrics, confirming the value of adapting to annotator-specific mask styles.

---

### 🏁 Official Challenge Results

Our final submission achieved the **6th best overall position** on the TrackRAD2025 leaderboard, with top-tier performance in multiple metrics:

| Rank | DSC (↑)      | CD [mm] (↓) | Rel. D98 (↑) | SD95 [mm] (↓) | SDavg [mm] (↓) | Runtime [s] (↓) |
|------|--------------|-------------|--------------|----------------|----------------|-----------------|
| 6th  | **0.8794 (4)** | 2.3192 (7)  | **0.9363 (3)** | 5.6298 (6)     | 2.6091 (8)     | 0.0751 (12)     |

We achieved:
- **4th best Dice** among all teams
- **3rd best relative dose (D98)** accuracy

## 🚀 Inference instructions

### 1. Download the fine-tuned model

Automatically download from Hugging Face:

```bash
python download.py
```
This will create the following file: ./SAM2.1_b+_finetune.pt

🔗 Hugging Face model page: https://huggingface.co/VBoussot/Trackrad2025

### 2. Prepare your dataset

Expected folder structure:

```
./input/
├── images/
│   ├── mri-linacs/
│   │   └── image.mha
│   └── mri-linac-target/
│       └── mask.mha
```
You can modify inference.py if your file names differ.

### 3. Run inference

```bash
python inference.py
```

Predictions will be saved in ./output/.
