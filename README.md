# Benchmarking Medical Vision-Language Models for Radiology Image Captioning

**Course:** Artificial Intelligence | **Faculty of Applied Sciences, UCU (2026)**

## Project Overview
This project focuses on a comparative analysis and benchmarking of multimodal Vision-Language Models (VLMs) for automatic caption generation of radiological images. The primary objective is to investigate whether augmenting prompts with predicted medical concepts (UMLS CUI) improves caption quality, serving as a digital assistant for radiologists.

## Dataset: ROCOv2
The project utilizes the open-source **ROCOv2 (Radiology Objects in Context v2)** dataset:
* **Total Size:** 79,789 radiological images.
* **Train Set:** 59,958 images.
* **Validation Set:** 9,904 images.
* **Test Set:** 9,927 images.
* **Annotations:** UMLS CUI medical concepts and natural language captions.

## Current Progress (Midterm)

The following modules and tasks have been implemented up to the midterm stage:

### 1. Exploratory Data Analysis (EDA) and Preprocessing
* Performed exploratory data analysis on class distributions, modalities, and text captions in the ROCOv2 dataset.

### 2. Baseline Medical Concept Detector (ResNet-50)
* **Architecture:** Pretrained ImageNet `ResNet-50`.
* **Task:** Multi-label concept classification across the full vocabulary of **1,947 UMLS CUIs**.
* **Loss Function:** **Asymmetric Loss** to handle severe class imbalance.
* **Augmentations:** Applied image rotation and zoom augmentations.
* **Tuning:** Tuned global prediction threshold on the validation set to maximize F1-score.

### 3. Baseline VLM Inference
Configured and executed baseline caption generation for **4 medical Vision-Language Models**:
* **MedGemma**
* **LLaVA-Med v1.5**
* **MedMO-4B-Next**
* **Hulu-Med-4B**

Generated baseline captions using a fixed instruction prompt without concept context.

### 4. Automated Evaluation Pipeline
Built an automated evaluation pipeline to compute quantitative caption quality metrics:
* **BLEU (1–4):** Lexical precision across n-grams.
* **ROUGE-L:** Sentence structure and longest common subsequence.
* **CIDEr:** Importance of rare clinical findings via TF-IDF weighting.
* **BERTScore:** Semantic similarity based on contextual embeddings.

## Contribution Statement (Completed Work)

* **Viktoria Kukurudza (50%):** Performed exploratory data analysis and preprocessing of the ROCOv2 dataset; developed, augmented, and trained the ResNet-50 concept detector on 1,947 UMLS CUIs using Asymmetric Loss and F1 threshold tuning; prepared project documentation.
* **Mariia Krasniukevych (50%):** Configured and executed inference for the four VLM models (MedGemma, LLaVA-Med v1.5, MedMO-4B-Next, Hulu-Med-4B); built the automated generation quality evaluation pipeline (BLEU-1-4, ROUGE-L, CIDEr, BERTScore).

**Mentor:** Yaroslav Prytula