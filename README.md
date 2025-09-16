[![Grand Challenge](https://img.shields.io/badge/Grand%20Challenge-TrackRad_2025-blue)](https://trackrad2025.grand-challenge.org/) [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-TrackRAD_2025-orange)](https://huggingface.co/VBoussot/Trackrad2025)

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

#### 📌 Effect of fine-tuning (on SAM 2.1 b+):

| Metric         | SAM 2.1 b+ | Ensemble prompt | Fine-tuned SAM 2.1 b+ |
|----------------|-----------|------------------|------------------------|
| **DSC**        | 0.916     | 0.908            | **0.929**              |
| **HD (mm)**    | 3.26      | 3.51             | **2.78**               |
| **ASD (mm)**   | 1.20      | 1.29             | **1.00**               |
| **CD (mm)**    | 1.34      | 1.39             | **1.16**               |

Fine-tuning the SAM 2.1 l model improved all metrics, confirming the value of adapting to annotator-specific mask styles.

---

### 🔬 Preliminary Test Set Evaluation

We compared the performance of **SAM 2.1 b+ with and without fine-tuning** on **unseen preliminary test data** from TrackRAD2025:

| Model                   | DSC (↑)        | CD [mm] (↓) | Rel. D98 (↑)   | SD95 [mm] (↓) | SDavg [mm] (↓) |
|-------------------------|----------------|-------------|----------------|----------------|----------------|
| SAM 2.1 b+ (fine-tuned) | **0.9152**| **1.4912**  | **0.9803**  | **4.2141**      | **1.5008**      |
| SAM 2.1 b+ (no fine-tune)| 0.9039     | 1.8864      | 0.9312      | 4.9406      | 1.7000      |

👉 Fine-tuning on the annotated TrackRAD2025 data significantly improved segmentation and tracking performance.

---

### 🏁 Official Challenge Results

Our final submission achieved the **6th best overall position** on the TrackRAD2025 leaderboard, with top-tier performance in multiple metrics:

| Rank | DSC (↑)      | CD [mm] (↓) | Rel. D98 (↑) | SD95 [mm] (↓) | SDavg [mm] (↓) | Runtime [s] (↓) |
|------|--------------|-------------|--------------|----------------|----------------|-----------------|
| 6th  | **0.8794 (4)** | 2.3192 (7)  | **0.9363 (3)** | 5.6298 (6)     | 2.6091 (8)     | 0.0751 (12)     |

We achieved:
- **4th best Dice** among all teams
- **3rd best relative dose (D98)** accuracy

🔗 [View the full TrackRAD2025 leaderboard](https://trackrad2025.grand-challenge.org/evaluation/final-testing/leaderboard/)

## 🚀 Inference Instructions

### 1. Download the fine-tuned model

Automatically download the fine-tuned SAM model from Hugging Face:

```bash
python download.py
```

This will create the following file:

```
./SAM2.1_b+_finetune.pt
```

🔗 [Model on Hugging Face](https://huggingface.co/VBoussot/Trackrad2025)

---

### 2. Prepare your dataset

You can prepare the dataset manually or automatically.

#### ✅ Option 1 – Manual structure

Expected folder structure:

```
./Dataset/
├── A_001/
│   ├── IMAGE.mha
│   └── MASK.mha
├── A_002/
│   ├── IMAGE.mha
│   └── MASK.mha
└── ...
```

- `IMAGE.mha` is the input cine-MRI frame  
- `MASK.mha` is the binary segmentation mask of the first annotated frame

#### ⚙️ Option 2 – Automatic generation

Run the following command to automatically convert the TrackRAD2025 labeled dataset into the proper format:

```bash
python data.py
```

This will generate the `./Dataset/` folder with one subfolder per patient (`A_001/`, etc.), ready for training or inference.

---

### 3. Run inference

Use the following command to run inference on the prepared dataset:

```bash
python model.py
```

Predictions will be saved as:

```
./Dataset/A_XXX/SAM21FINETUNE.mha
```

---

### 4. Run evaluation

To evaluate the predictions against the ground truth masks:

```bash
konfai EVALUATION -y
```

---

## 📚 References

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., *et al.* (2023).  
  **Segment Anything**. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4015–4026.  
  [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

- Ma, J., He, Y., Li, F., Han, L., You, C., & Wang, B. (2024).  
  **Segment Anything in Medical Images**. *Nature Communications*, 15, 654.  
  [https://www.nature.com/articles/s41467-024-44979-3](https://www.nature.com/articles/s41467-024-44979-3)

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**. *arXiv preprint arXiv:2508.09823*.  
  [https://arxiv.org/abs/2508.09823](https://arxiv.org/abs/2508.09823)


