# Initial Test of Transform-Dependent Attacks
**Dataset:** `imagenet_compatible` (Jsut the first 100 images)
**Transforms Evaluated:** Scaling (0.5x), Gamma (0.5), Blurring (Sigma 1.5)

## 1. Experimental Setup
The model was run using `inception` as the surrogate model with the following hyperparameters:

```bash
--model_name "inception" \
--pretrained_diffusion_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
--attack_mode "transform_dependent" \
--attack_loss_weight 10 \
--cross_attn_loss_weight 10000 \
--self_attn_loss_weight 100 \
--diffusion_steps 20 \
--start_step 15 \
--iterations 30 \
--guidance 2.5 \
--attack_loss_type "ce"
```

### Model Performance
**Benign Consistency** (Clean Accuracy) and **Attack Success** (Adversarial Accuracy) across the Scaling, Gamma, and Blurring.

Note that the adversarial examples were made with the **Inception** model (bolded below). 

| Target Model | Scaling Clean % | Scaling Adv % | Gamma Clean % | Gamma Adv % | Blur Clean % | Blur Adv % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet** | 89.00 | 0.00 | 84.00 | 0.00 | 89.00 | 0.00 |
| **VGG** | 79.00 | 0.00 | 85.00 | 0.00 | 76.00 | 0.00 |
| **Mobile** | 79.00 | 0.00 | 74.00 | 0.00 | 71.00 | 0.00 |
| **Inception** | **97.00** | **5.00** | **100.00** | **59.00** | **100.00** | **72.00** |
| **ConvNext** | 93.00 | 0.00 | 96.00 | 0.00 | 92.00 | 0.00 |
| **ViT** | 87.00 | 0.00 | 89.00 | 0.00 | 87.00 | 0.00 |
| **Swin** | 87.00 | 0.00 | 94.00 | 0.00 | 95.00 | 0.00 |
| **DeiT-B** | 86.00 | 0.00 | 89.00 | 0.00 | 93.00 | 0.00 |
| **DeiT-S** | 85.00 | 0.00 | 87.00 | 0.00 | 92.00 | 0.00 |
| **Mixer-B** | 72.00 | 0.00 | 75.00 | 0.00 | 74.00 | 0.00 |
| **Mixer-L** | 67.00 | 0.00 | 73.00 | 0.00 | 68.00 | 0.00 |
| **Adv Inception v3** | 79.00 | 0.00 | 75.00 | 0.00 | 77.00 | 2.00 |
| **Ens3 Adv Inc v3** | 62.00 | 0.00 | 66.00 | 0.00 | 60.00 | 0.00 |
| **Ens4 Adv Inc v3** | 71.00 | 0.00 | 67.00 | 0.00 | 64.00 | 1.00 |
| **Ens Adv Inc Res v2**| 88.00 | 0.00 | 85.00 | 0.00 | 82.00 | 1.00 |

### Image Quality Metrics

| Experiment | Parameter | LPIPS Score | FID Score |
| :--- | :--- | :--- | :--- |
| **Scaling** | Scale 0.5x | 0.1432 | 206.10 |
| **Gamma** | Gamma 0.5 | 0.1233 | 208.27 |
| **Blurring** | Blur Sigma 1.5 | 0.1291 | 205.30 |

### Analysis:

Since these attacks were generated using Inception’s predictions, I expected that model to show the best performance. (Which it does)

First off, the transferability of the perturbations is basically zero. While the images themselves remain benign across different architectures, the hidden adversarial trigger completely fails to activate on anything other than the source model.

The scaling transform is particularly terrible. Even on the source Inception model, it only succeeded 5% of the time. I’m  not sure why it did so bad comapred to the other transforms.

Looking at the big picture, the overall attack success rate is pretty underwhelming. We might need to adjust our approach (tweaking the loss weights, increasing the optimization steps, shifting the optimization location, etc). It might be worth sacrificing a tiny bit of imperceptibility if it buys us a better success rate.

A few notes on the setup and metrics:

- Sample Size: We only tested 100 images, so it’s a small batch.

- Compute: For context, each transform took about 2.5 hours on an L4 GPU.

- Perceptual Metrics: I still need to visually inspect the results, but the numbers give us a hint:
  - LPIPS (0.12–0.14): This is right on par with the attacks in DiffAttack.
  - FID: This score is massive, but we can ignore it. FID is  unreliable with a small sample size (100 images), so it's not giving us a true reading of quality.