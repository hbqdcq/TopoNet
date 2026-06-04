# Topology-Guided Spatial Representation Learning for Robust Volumetric Medical Image Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20139085.svg)](https://doi.org/10.5281/zenodo.20139085)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


* Open-sourced the core implementation and archived via Zenodo with DOI: `10.5281/zenodo.20139085`.


---

## 🌟 Introduction
Volumetric medical image classification remains challenging due to limited robustness and interpretability, as modern models rely mainly on appearance cues and fail to capture global anatomical topology.

This work presents a **topology-guided dual-branch framework** that integrates multiscale topological structures into volumetric representation learning.

* **Cubical Persistent Homology:** Used to extract robust topological features while suppressing noise via statistical confidence bands.
* **Topology-to-Voxel Mapping:** Resolves the spatial detachment issue in traditional methods by restoring spatial information.
* **Cross-Branch Attention:** Fuses image and topology modalities via stage-wise interaction.

---

## 🏗️ Repository Structure
```text
TopoNet/
├── data/                  # Directory for NIfTI datasets and processed ROIs
├── networks/              # Dual-branch network, Cross-Branch Attention, ViT modules
├── utils/                 
├── train.py               # Main training pipeline
├── test.py                # Evaluation and inference script
├── requirements.txt       # Environment dependencies
└── README.md
