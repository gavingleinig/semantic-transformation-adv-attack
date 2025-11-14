# Transformation-Triggered Adversarial Attacks

This repository contains code for exploring **conditional generative adversarial examples**, adapted from the codebase of "Semantic Adversarial Attacks via Diffusion Models" [link](https://github.com/steven202/semantic_adv_via_dm#)

## Project Overview

The core project idea is to explore transformation-triggered natural adversarial examples.

Unlike a standard semantic attack where the image is always adversarial, this project focuses on creating images that appear benign to a classifier under normal viewing conditions. The adversarial property is "hidden" and only activated when the image is subjected to a specific, physically plausible transformation.

The "generative" aspect involves using models (like diffusion) to create or semantically modify an image to embed this conditional vulnerability. The "transformation" trigger could be a common, real-world manipulation, such as:

* Rotation (e.g., 90° or 180°)
* Viewing through a colored or polarized filter
* Zooming in or out

### Example Attack Vector

An attacker creates an image for a digital billboard. Under normal operation, a classifier (e.g., a self-driving car’s sensor) views it and correctly classifies it as a benign advertisement. Later, the attacker (or a compromised system) rotates the on-screen image 180°. Due to the embedded adversarial trigger, the rotated image is now misclassified as a malicious target, such as a stop sign.

### Methodology

This repository uses the optimization loop and generative capabilities of the original codebase to implement a new loss function.

---

## Setup

### Environment Setup

We use Conda to manage the environment. You can create and activate the environment by running:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate semantic_adv
```


### Pre-trained Models

This method uses several pre-trained models used by the diffusion framework.

1.  **Classifier Models:** Download the pre-trained target classifiers and place them in the `classifier_ckpts/` directory as specified in `configs/paths_config.py`.
2.  **Diffusion Models:** Download the pre-trained diffusion models and place them in the `pretrained_models/` directory, updating the paths in `configs/paths_config.py`.


---

## How to Run

The core attack logic is implemented in `diffusionclip.py` in the `generate_attack` method. This project's goal is achieved by modifying the loss function within this method.

### Attack Execution

The main entry point is `main.py`. The command-line arguments allow you to specify the configuration file, dataset, and attack parameters.

A typical command to run an attack looks like this:

```python
python main.py --config [config_file.yml] --attack --n_test_img 100 --t_0 400 --n_inv_step 40 --n_train_step 10 --bs_test 1 --lr_clip_lat_opt 0.05 --clip_model_name "ViT-B/16" --edit_attr [your_attack_name]
```

### Modifying the Loss Function

To implement the conditional attack, you will need to modify the loss calculation inside `diffusionclip.py`, specifically within the `generate_attack` method.

The new loss function is a weighted sum of:

1.  **Benign Loss:** `ClassificationLoss(Classifier(x_A), Original_Label)`
2.  **Adversarial Loss:** `ClassificationLoss(Classifier(T(x_A)), Target_Label)`
3.  **Perceptual/ID Loss:** `LPIPS(x_A, x_Original)` or `IDLoss(x_A, x_Original)` (to ensure the image still looks like the original).

## Code Structure
```bash
├── configs/                # Configuration files (.yml) and path definitions (paths_config.py)
├── commands/               # Example shell scripts for running experiments
├── datasets/               # PyTorch Dataset classes for CelebA-HQ, AFHQ, etc.
├── losses/                 # Loss functions (CLIPLoss, IDLoss).
│   ├── clip_loss.py
│   └── id_loss.py
├── models/                 # Core models
│   ├── ddpm/                 # Denoising Diffusion Probabilistic Models
│   ├── improved_ddpm/        # Improved DDPM implementation
│   └── insight_face/         # ArcFace model for ID loss
├── saliency/               # Saliency map methods (GradCAM, FullGrad)
├── utils/                  # Utility scripts
├── main.py                 # Main script to run attacks and experiments
├── diffusionclip.py        # Core logic for the DiffusionCLIP attack method
└── README.md               # This README file
```