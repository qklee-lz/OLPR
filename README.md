# Oral Leukoplakia Progression Recognition (OLPR)

This repository contains the official implementation of our paper:

**Title:** Advancing Oral Leukoplakia Progression Recognition: A benchmark with Dataset, Method, and Application 
**Journal:** Under Review on ***Neural Networks***

---

## 🔍 Overview
In this paper, we introduce Oral Leukoplakia Progression Recognition (OLPR) as a novel benchmark task to classify oral lesions into three clinically relevant categories: normal, leukoplakia, and leukoplakia with cancer. We construct the OLPR dataset, a high-quality, annotated collection derived from multiple public datasets, and establish an external validation dataset using university and clinical data. 

---


## 📂 Data Access

The dataset is publicly available on Google Drive:

👉 [Google Drive Download Link](https://drive.google.com/file/d/1rRmZIiemByJr4RlXGqIzwu3-R6MBq1ug/view?usp=sharing)

---

## 📦 Dataset Structure

After downloading and extracting the dataset, the structure is as follows:
```
./root_data/
│
├── Normal/
│ ├── xxx.jpeg
│ ├── ...
├── Leukoplakia/
│ ├── xxx.jpeg
│ ├── ...
├── Leukoplakia_Cancer/
│ ├── xxx.jpeg
│ ├── ...
├── train.json
├── valid.json
└── test.json
```

It contains three categories:
| Category        | Samples |
|-----------------|---------|
| Normal          | 488     |
| Leukoplakia     | 132     |
| Leuk. Cancer    | 158     |

`train.json`, `val.json`, `test.json`: JSON files mapping image paths to labels  
  - **0 = Normal**
  - **1 = Leukoplakia**
  - **2 = Leukoplakia Cancer**


---

## 📊 Dataset Split

- **Training set**: 537 images  
- **Validation set**: 90 images  
- **Test set**: 151 images  


---


## 🧪 Benchmark with TorchVision & timm

To facilitate reproducibility, we provide benchmark baselines using models from **[torchvision](https://pytorch.org/vision/stable/models.html)** and **[timm](https://github.com/huggingface/pytorch-image-models)**.  



We provide unified training and evaluation scripts so that you can easily switch between different models (e.g., ResNet, DeiT, Swin Transformer V2, etc.) with only a command-line argument.  

### 🔹 Training
Example: train a **DeiT** model for 50 epochs
```bash
python train.py \
    --model deit_base_distilled_patch16_224 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --data_path ./root_data/
```
You can replace --model with any supported architecture name from torchvision or timm
(e.g., resnet50, vit_b_16, swinv2_base).

### 🔹 Evaluation
After training, evaluate a model checkpoint on the test set:
```bash
python evaluate.py \
    --model deit_base_distilled_patch16_224 \
    --weights checkpoints/best_model.pth \
    --data_path ./root_data/
```
You can replace --model with any supported architecture name from torchvision or timm
(e.g., resnet50, vit_b_16, swinv2_base).


### 🔹Additional Info
The OLPR-Net framework is based on our previous work (environment code base) published in [*IEEE JBHI CDTM*](https://github.com/qklee-lz/CDTM). The pre-trained weights will be released together with the acceptance of our paper.

### 🔹 Notes
```
--model : choose any available model name from TorchVision or timm.
--data_path : root directory containing your dataset (train.json, valid.json, test.json).
--weights : path to a trained checkpoint (default: checkpoints/best_model.pth).
Metrics reported: Loss, Accuracy, F1-score, Precision, Recall, Mean Accuracy, Per-class Accuracy.
```