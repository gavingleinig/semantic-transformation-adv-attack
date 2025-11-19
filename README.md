# Transformation-Triggered Adversarial Attacks

This repository contains code for exploring **conditional generative adversarial examples**, adapted from the codebase of "Diffusion Models for Imperceptible and Transferable Adversarial Attack" [link](https://github.com/WindVChen/DiffAttack)

Building off on the DiffAttack framework, our main contribution is to test and evaluable the feasability of transformation-triggered attacks.

Notes to team members!
- [main.py](main.py) parses args
- [diff_latent_attack.py](diff_latent_attack.py) runs code to make attacks adversarial
   - So, our main changes will be in diff_latent_attack.py
   - [DiffAttack](https://ieeexplore.ieee.org/abstract/document/10716799) uses 3 loss terms: 
      - L_attack, L_structure, and L_transfer
   - Most of the code in diff_latent_attack.py deals with these terms
   - For a basic attack to work, all that we really need is L_attack
- For our attack, I propose we modify the L_attack 

Link to Colab Notebook that sets environment up!
[link](https://colab.research.google.com/drive/16HFnYTZ5P4cJd80BqmL0yZq3HP7wpxVr?usp=sharing)

## Project Overview

The core project idea is to explore transformation-triggered natural adversarial examples.

Unlike a standard semantic attack where the image is always adversarial, this project focuses on creating images that appear benign to a classifier under normal viewing conditions. The adversarial property is "hidden" and only activated when the image is subjected to a specific, physically plausible transformation.

The "generative" aspect involves using models (like diffusion) to semantically modify an image to embed this conditional vulnerability. The "transformation" trigger could be a common, real-world manipulation, such as:

* Rotation (e.g., 90° or 180°)
* Viewing through a colored or polarized filter
* Zooming in or out

### Example Attack Vector

An attacker creates an image for a digital billboard. Under normal operation, a classifier (e.g., a self-driving car’s sensor) views it and correctly classifies it as a benign advertisement. Later, the attacker (or a compromised system) rotates the on-screen image 180°. Due to the embedded adversarial trigger, the rotated image is now misclassified as a malicious target, such as a stop sign.

### Methodology

This repository uses the optimization loop and generative capabilities of the original codebase (DiffAttack) to implement a new transformational trigger by a custom loss function.

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
   - There is current a demo-datasets in [demo](demo), you can directly run the optimization code below to see the results.
   - If you want to test the full `ImageNet-Compatible` dataset, please download the dataset [ImageNet-Compatible](https://drive.google.com/file/d/1sAD1aVLUsgao1X-mu6PwcBL8s68dm5U9/view?usp=sharing) and then change the settings of `--images_root` and `--label_path` in [main.py](main.py)

4. Pre-trained Models
   - We adopt `Stable Diffusion 1.5` as our diffusion model, you can load the pretrained weight by setting `--pretrained_diffusion_path="stable-diffusion-v1-5/stable-diffusion-v1-5"` in [main.py](main.py).
   - For the pretrained weights of the adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>) can be download them from [here](https://github.com/ylhz/tf_to_pytorch_model) and placed them into the directory `pretrained_models`.

```
   (Supplement done in DiffAttack) Attack **CUB_200_2011** and **Standford Cars** datasets
   - Dataset: Aligned with **ImageNet-Compatible**, we randomly select 1K images from **CUB_200_2011** and **Standford Cars** datasets, respectively. You can download the dataset here [[CUB_200_2011](https://drive.google.com/file/d/1umBxwhRz6PIG6cli40Fc0pAFl2DFu9WQ/view?usp=sharing) | [Standford Cars](https://drive.google.com/file/d/1FiH98QyyM9YQ70PPJD4-CqOBZAIMlWJL/view?usp=sharing)] and then change the settings of `--images_root` and `--label_path` in [main.py](main.py). Note that you should also set `--dataset_name` to `cub_200_2011` or `standford_car` when running the code.
   - Pre-trained Models: You can download models (ResNet50, SENet154, and SE-ResNet101) pretrained on CUB_200_2011 and Standford Cars from [Beyond-ImageNet-Attack](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack) repository. Then place them into the directory `pretrained_models`.
```

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

Our project tests our attacks against adversarially trained models.  Future work can evaluate our attack's power to deceive other defensive approaches. Other potential defeneses to try are as follows:
- [HGD](https://github.com/lfz/Guided-Denoise): Change the input size to 224, and then directly run the original code.
- [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense): Since our target size is 224, we reset the image scale augmentation proportionally (232~248). Then run the original code.
- [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): Since its ensembled models failed to process inputs with 224 size, we run its original code that resized the inputs to 299 size.
- [RS](https://github.com/locuslab/smoothing): Change the input size to 224 and set sigma=0.25, skip=1, max=-1, N0=100, N=100, alpha=0.001, then run the original code.
- [NRP](https://github.com/Muzammal-Naseer/NRP): Change the input size to 224 and set purifier=NRP, dynamic=True, then run the original code.
- [DiffPure](https://github.com/NVlabs/DiffPure): Modify the original codes to evaluate the existing adversarial examples, not crafted examples again.

## Results

