# MADation: Face Morphing Attack Detection with CLIP and LoRA

**Using the CLIP AI model and LoRA to efficiently detect face morphing attacks.**

This repository is an implementation of the deep learning project "Face Morphing Attack Detection," based on the paper *MADation: Face Morphing Attack Detection with Foundation Models*.

[![Paper](https://img.shields.io/badge/arXiv-2501.03800-b31b1b.svg)](https://arxiv.org/pdf/2501.03800v3)

## 📖 Table of Contents
* [1. Introduction: The Problem](#1-introduction-the-problem)
* [2. Solution: MADation](#2-madation)
* [3. Methodologies Explored](#3-methodologies-explored)
* [4. Ablation Study](#4-ablation-study)

## 1. Introduction: The Problem

Modern Face Recognition (FR) systems are highly advanced, but they remain vulnerable to a significant security threat: **Morphing Attacks (MA)**.

A Morphing Attack involves fusing the identity information from two or more people into a single, synthetic image. This "morphed" image is designed to be verifiable for all contributing individuals, posing a major risk to security systems like automated border control, where multiple people could use the same passport.

This project implements and evaluates a **Single-Image Morphing Attack Detection (MAD)** system, which is designed to distinguish these malicious morphed images from legitimate (bona-fide) photos.

## 2. MADation

While large-scale Foundation Models (FMs) like CLIP have powerful generalization capabilities, their use in highly specific, domain-specific tasks like MAD is not well-explored.

**MADation**, is a framework that efficiently adapts a pre-trained FM (CLIP) for the MAD task.

Instead of retraining the entire, massive model (which is computationally expensive and requires huge datasets), MADation uses **Low-Rank Adaptation (LoRA)**.

* **How it works:** LoRA **freezes** the original CLIP model weights.
* It then injects small, trainable low-rank matrices into the Multi-Headed Self-Attention (MSA) blocks of the model.
* Only these small matrices (and a new binary classification layer) are trained.
* This efficiently "shifts" the model's feature space to be highly effective for MAD, combining the power of pre-training with task-specific fine-tuning.

## 3. Methodologies Explored

To prove the effectiveness of MADation, we implemented and compared it against three key baselines.

1.  **`MADation`:**
    * **Architecture:** Pre-trained CLIP (ViT) model + LoRA adapters.
    * **Training:** LoRA layers and a final classification layer are trained simultaneously.

2.  **`FE` (Feature Extractor):**
    * **Architecture:** Pre-trained CLIP (ViT) model + a classification layer.
    * **Training:** The *entire* CLIP model is **frozen**. Only the final classification layer is trained. This tests the raw, unadapted features of CLIP.

3.  **`ViT-FS` (ViT From Scratch):**
    * **Architecture:** The same ViT architecture as CLIP, but with no pre-training.
    * **Training:** All model parameters are initialized randomly and trained from scratch on *only* the MAD dataset. This tests the value of pre-training.

4.  **`TI` (Text-Image Zero-Shot):**
    * **Architecture:** The standard, off-the-shelf, frozen CLIP model.
    * **Training:** No training is performed. Classification is done by comparing the image's embedding to the text embeddings of "a face morphing attack" and "a bona-fide presentation".

## 4. Ablation Study

All models were trained on the SMDD dataset and evaluated on the MorDIFF dataset. The results clearly demonstrate the superiority of the MADation framework.

| Method | EER (%) | BPCER (%) @ 1% APCER | APCER (%) @ 1% BPCER |
| :--- | :---: | :---: | :---: |
| TI (Zero-Shot) | 51.90 | 99.02 | 100.00 |
| ViT-FS (From Scratch) | 28.14 | 34.37 | 100.00 |
| FE (Frozen Extractor) | 17.86 | 12.62 | 59.22 |
| **MADation (Ours)** | **1.10** | **0.00** | **1.94** |
*<small>Metrics: **EER** (Equal Error Rate), **BPCER** (Bona-fide Presentation Classification Error Rate), **APCER** (Attack Presentation Classification Error Rate). Lower is better for all metrics.</small>*

### Analysis of Results

* **`TI` (Zero-Shot)** performed close to random guessing (50%). This shows that MAD is too domain-specific for a generic, unadapted CLIP model.
* **`ViT-FS` (From Scratch)** performed very poorly. This confirms that large transformer models like ViT require massive pre-training, and a small, specific dataset like SMDD is not enough.
* **`FE` (Frozen Extractor)** was significantly better, proving that the pre-trained CLIP features are a good starting point. However, its high error rate shows the features are not perfectly suited for MAD without adaptation.
* **`MADation` (Ours)** was the clear winner, with an EER of only 1.10%. This validates our approach: it successfully combines the powerful general features from pre-training with efficient, task-specific adaptation using LoRA.
