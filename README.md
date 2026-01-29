## CSIRO Image2Biomass: Multi-Stage Biomass Prediction
LINK: https://www.kaggle.com/competitions/csiro-biomass

<p align="center">
  <img src="assets/thumbnail_biomass.png" width="700">
</p>

This repository contains the implementation of two deep learning architectures developed for the **CSIRO Image2Biomass Prediction** challenge. The project focuses on estimating five biomass components—**Dry Clover, Dry Dead, Dry Green, Dry Total, and GDM (Green Dry Matter)**—directly from RGB field imagery.

---

## Model Architectures & Logic

### Model 1: DINOv2 Hierarchical Regressor (Baseline)

Model 1 establishes a robust baseline by leveraging Self-Supervised Learning (SSL) features for high-resolution regression.

  * **Backbone:** `dinov2_vits14`. The model utilizes a frozen backbone with the **last 4 blocks unfrozen** to specialize the features for agricultural textures and fine-grained leaf details.
  * **Architectural Logic:** A 384-dimensional `[CLS]` token is passed through a hierarchical MLP head with `LayerNorm` and `GELU` activations. High dropout (0.4) is applied to prevent overfitting on the limited image set.
  * **Target Strategy:**
    * **Log-Normal Scaling:** Targets are transformed via  and Z-score normalized to stabilize training gradients.
    * **Weighted MSE Loss:** Prioritizes `Dry_Total_g` (0.5 weight) and `GDM_g` (0.2 weight) to align with competition scoring priorities.
  
  
  * **Input Pipeline:**  RGB images with standard color jitter, Gaussian blur, and discrete  rotations.

---

### Model 2: DINOv3 + Teacher-Student Distillation (Optimized)

Model 2 evolves the approach into a multi-modal, multi-scale framework that incorporates domain-specific metadata and spatial attention logic.

* **Multi-Scale Attention Fusion:**
  * **Global + Local Crops:** Simultaneously processes a global image and **4 high-resolution corner crops**.
  * **Crop Attention Head:** A dedicated head calculates importance weights for each crop, allowing the model to focus on the most biomass-dense regions before final feature fusion.

## Citation

**BibTeX:**

```bibtex
@misc{kunwar2026csirobiomass,
  author = {Kunwar, Mayank},
  title = {CSIRO - Image2Biomass Prediction: Optimized DINOv3 with Teacher-Student Distillation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/mayankkunwar93/CSIRO---Image2Biomass-Prediction}}
}

```

**APA:**

Kunwar, M. (2026). *CSIRO - Image2Biomass Prediction: Optimized DINOv3 with Teacher-Student Distillation*. GitHub Repository. [https://github.com/mayankkunwar93/CSIRO---Image2Biomass-Prediction](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/mayankkunwar93/CSIRO---Image2Biomass-Prediction)

* **Teacher-Student Knowledge Distillation:**
  * An auxiliary **Teacher NN** is trained on non-image metadata: **NDVI, Plant Height, Seasonality (Month), and Species Taxonomy**.
  * The image-based Student model is supervised by both ground truth and Teacher predictions, effectively grounding visual features in biological reality.


* **Taxonomic Feature Engineering:**
The model utilizes a custom hierarchical mapping to transform 15+ species labels into a structured multi-hot vector. This allows the model to share weights across similar plant functional types:
  * **Grassy types:** Anchored with a "grass" tag (e.g., BarleyGrass, Ryegrass).
  * **Clovers:** Influence the specific Clover prediction head (e.g., Subclover, WhiteClover).
  * **Legumes & Weeds:** Categorized into "legume" or "broadleaf" to account for different leaf-to-stem ratios.


* **Biological Logic & Loss:**
  * **Ratio Protection:** Uses **KL Divergence** to enforce realistic proportions between components (e.g., ensuring GDM and Dead components logically sum toward the Total).
  * **Weighted Huber Loss:** Replaces MSE with Huber loss to provide better robustness against outliers in the biomass measurements.


---

### Species to Subspecies Mapping Logic

For transparency, the following taxonomy was used to enrich the metadata features:

| Category | Subspecies Tags |
| --- | --- |
| **Grasses** | Barley, Brome, Fescue, Phalaris, Rye, Silver, Spear |
| **Clovers** | White, Subclover, Dalkeith, Losa |
| **Legumes** | Lucerne |
| **Broadleaf** | Capeweed, Crumbweed |


---

## Performance Comparison

| Feature | Model 1 (Baseline) | Model 2 (Optimized) |
| --- | --- | --- |
| **Backbone** | DINOv2-S | **DINOv3 w/ Attention Pooling** |
| **View Strategy** | Single Global View | **Global + 4 Attentive Crops** |
| **Supervision** | Ground Truth Only | **Teacher Distillation + Metadata** |
| **Target Logic** | Independent Regression | **Hierarchical Ratio Reconstruction** |
| **Loss Function** | Weighted MSE | **Huber + KL Divergence** |
| **Optimization** | Cosine Annealing | **Differential Learning Rates** |

---

## Project Structure

The repository is organized into two distinct experimental versions:

`src/v1_dinov2_baseline/`
Standard DINOv2 implementation used to establish performance benchmarks.

  * `train.py`: Main script for K-Fold baseline training.
  * `models.py`: DINOv2 backbone with a hierarchical MLP regression head.
  * `dataset.py`: Single-view image loading and standard augmentations.
  * `utils.py`: Weighted MSE loss and baseline R2 metrics.

`src/v2_dinov3_optimized/`
Proposed DINOv3 architecture with Multi-Scale Attention and Knowledge Distillation.

  * `train.py`: Coordinates the Teacher-Student training loop and metadata integration.
  * `models.py`: Features the DINOv3 backbone, Attention Pooling, and Teacher NN architecture.
  * `dataset.py`: Implements Global + 4-Corner Crop logic for high-resolution feature extraction.
  * `utils.py`: Contains advanced Taxonomic Mapping and the Huber + KL Divergence loss function.

---


## Data Attribution
 The dataset for this project is sourced from the CSIRO - Image2Biomass Prediction competition on Kaggle.

Source: https://www.kaggle.com/competitions/csiro-biomass

License: Access to this data is governed by the CSIRO competition rules. Users must download the data directly from Kaggle to run the scripts in this repository.
