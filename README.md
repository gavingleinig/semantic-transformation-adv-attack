# Transformation-Triggered Adversarial Attacks

This repository contains code for exploring **conditional generative adversarial examples**, adapted from the codebase of "Diffusion Models for Imperceptible and Transferable Adversarial Attack" [link](https://github.com/WindVChen/DiffAttack)

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

## Requirements

1. Hardware Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 16GB memory

2. Software Requirements
    - Python: 3.8
    - CUDA: 11.3
    - cuDNN: 8.4.1

   To install other requirements:

   ```
   pip install -r requirements.txt
   ```

3. Datasets
   - There have been demo-datasets in [demo](demo), you can directly run the optimization code below to see the results.
   - If you want to test the full `ImageNet-Compatible` dataset, please download the dataset [ImageNet-Compatible](https://drive.google.com/file/d/1sAD1aVLUsgao1X-mu6PwcBL8s68dm5U9/view?usp=sharing) and then change the settings of `--images_root` and `--label_path` in [main.py](main.py)

4. Pre-trained Models
   - We adopt `Stable Diffusion 2.0` as our diffusion model, you can load the pretrained weight by setting `--pretrained_diffusion_path="stabilityai/stable-diffusion-2-base"` in [main.py](main.py).
   - For the pretrained weights of the adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>) in Section 4.2.2 of our paper, you can download them from [here](https://github.com/ylhz/tf_to_pytorch_model) and then place them into the directory `pretrained_models`.

5. (Supplement) Attack **CUB_200_2011** and **Standford Cars** datasets
   - Dataset: Aligned with **ImageNet-Compatible**, we randomly select 1K images from **CUB_200_2011** and **Standford Cars** datasets, respectively. You can download the dataset here [[CUB_200_2011](https://drive.google.com/file/d/1umBxwhRz6PIG6cli40Fc0pAFl2DFu9WQ/view?usp=sharing) | [Standford Cars](https://drive.google.com/file/d/1FiH98QyyM9YQ70PPJD4-CqOBZAIMlWJL/view?usp=sharing)] and then change the settings of `--images_root` and `--label_path` in [main.py](main.py). Note that you should also set `--dataset_name` to `cub_200_2011` or `standford_car` when running the code.
   - Pre-trained Models: You can download models (ResNet50, SENet154, and SE-ResNet101) pretrained on CUB_200_2011 and Standford Cars from [Beyond-ImageNet-Attack](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack) repository. Then place them into the directory `pretrained_models`.

## Crafting Adversarial Examples

To craft adversarial examples, run this command:

```
python main.py --model_name <surrogate model> --save_dir <save path> --images_root <clean images' path> --label_path <clean images' label.txt>
```
The specific surrogate models we support can be found in `model_selection` function in [other_attacks.py](other_attacks.py). You can also leverage the parameter `--dataset_name` to generate adversarial examples on other datasets, such as `cub_200_2011` and `standford_car`.

The results will be saved in the directory `<save path>`, including adversarial examples, perturbations, original images, and logs.

For some specific images that distort too much, you can consider weaken the inversion strength by setting `--start_step` to a larger value, or leveraging pseudo masks by setting `--is_apply_mask=True`.


## Evaluation


### Robustness on other normally trained models

To evaluate the crafted adversarial examples on other black-box models, run:

```
python main.py --is_test True --save_dir <save path> --images_root <outputs' path> --label_path <clean images' label.txt>
```
The `--save_dir` here denotes the path to save only logs. The `--images_root` here should be set to the path of `--save_dir` in above [Crafting Adversarial Examples](#crafting-adversarial-examples).


### Robustness on defensive approaches

Apart from the adversarially trained models, we also evaluate our attack's power to deceive other defensive approaches as displayed in Section 4.2.2 in our paper, their implementations are as follows:
- Adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>): Run the code in [Robustness on other normally trained models](#robustness-on-other-normally-trained-models).
- [HGD](https://github.com/lfz/Guided-Denoise): Change the input size to 224, and then directly run the original code.
- [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense): Since our target size is 224, we reset the image scale augmentation proportionally (232~248). Then run the original code.
- [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): Since its ensembled models failed to process inputs with 224 size, we run its original code that resized the inputs to 299 size.
- [RS](https://github.com/locuslab/smoothing): Change the input size to 224 and set sigma=0.25, skip=1, max=-1, N0=100, N=100, alpha=0.001, then run the original code.
- [NRP](https://github.com/Muzammal-Naseer/NRP): Change the input size to 224 and set purifier=NRP, dynamic=True, then run the original code.
- [DiffPure](https://github.com/NVlabs/DiffPure): Modify the original codes to evaluate the existing adversarial examples, not crafted examples again.

## Results

