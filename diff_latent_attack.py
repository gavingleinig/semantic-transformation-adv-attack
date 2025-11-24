from typing import Optional
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import other_attacks
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def cw_loss_targeted(logits: torch.Tensor, target: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    """Targeted Carlini-Wagner style margin loss.
    logits: [B, C], target: [B] or [B, 1]
    This version ensures `target` is a long tensor on the same device as `logits`,
    and uses a fill value created from `logits` to avoid dtype/device mismatches
    when calling scatter_.
    """
    # allow target to be a column vector
    if target is None:
        raise ValueError("target must be provided for targeted CW loss")
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    if target.dim() == 2:
        target = target.squeeze(1)
    # ensure integer indices and same device as logits
    target = target.long().to(logits.device)
    # support scalar target or single-element target for multi-sample logits by broadcasting
    if target.dim() == 0:
        target = target.unsqueeze(0)
    if target.size(0) == 1 and logits.size(0) > 1:
        target = target.expand(logits.size(0))
    if target.size(0) != logits.size(0):
        raise ValueError("Batch size of `target` must match `logits`")

    # gather the logit of the target class
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    others = logits.clone()
    # create fill value on the same device/dtype as logits to avoid scatter_ errors
    fill_val = logits.new_tensor(-1e10)
    # scatter_ sometimes expects src to have the same number of dimensions as index.
    # Use advanced indexing assignment which correctly broadcasts a scalar fill value
    # to each row's target column and avoids dimension mismatch errors.
    batch = logits.size(0)
    row_idx = torch.arange(batch, device=logits.device)
    others[row_idx, target] = fill_val
    max_other = others.max(1).values
    loss = torch.clamp(max_other - target_logit + kappa, min=0.0)
    return loss.mean()


def cw_loss_untargeted(logits: torch.Tensor, true: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    """Untargeted CW loss: push true class logit below the highest other logit.
    Ensures `true` is long and on the same device as `logits`.
    """
    if true is None:
        raise ValueError("`true` labels must be provided for untargeted CW loss")
    if isinstance(true, np.ndarray):
        true = torch.from_numpy(true)
    if true.dim() == 2:
        true = true.squeeze(1)
    true = true.long().to(logits.device)
    # support scalar/1-element true labels by broadcasting to match logits batch
    if true.dim() == 0:
        true = true.unsqueeze(0)
    if true.size(0) == 1 and logits.size(0) > 1:
        true = true.expand(logits.size(0))
    if true.size(0) != logits.size(0):
        raise ValueError("Batch size of `true` must match `logits`")

    true_logit = logits.gather(1, true.unsqueeze(1)).squeeze(1)
    others = logits.clone()
    fill_val = logits.new_tensor(-1e10)
    # scatter_ sometimes expects src to have the same number of dimensions as index.
    # Use advanced indexing assignment which correctly broadcasts a scalar fill value
    # to each row's true column and avoids dimension mismatch errors.
    batch = logits.size(0)
    row_idx = torch.arange(batch, device=logits.device)
    others[row_idx, true] = fill_val
    max_other = others.max(1).values
    loss = torch.clamp(true_logit - max_other + kappa, min=0.0)
    return loss.mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation loss for images in [B, C, H, W] format."""
    if x.dim() != 4:
        raise ValueError("tv_loss expects a 4D tensor [B,C,H,W]")
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def transform_scale(img_tensor, scale_factor):
    # F.interpolate is differentiable
    scale_val = scale_factor.item()
    return F.interpolate(img_tensor, scale_factor=scale_val, mode='bilinear', align_corners=False)

def transform_identity(img_tensor, dummy_param):
    return img_tensor

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
           noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.enable_grad()
def diffattack(
        model,
        label,
        controller,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        model_name="inception",
        save_path=r"C:\Users\PC\Desktop\output",
        res=224,
        start_step=15,
        iterations=30,
        verbose=True,
        topN=1,
        args=None
):
    if args.dataset_name == "imagenet_compatible":
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    _, pred_labels = pred.topk(topN, largest=True, sorted=True)

    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)])
    prompt = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2
    print("prompt generate: ", prompt[0], "\tlabels: ", pred_labels.cpu().numpy().tolist())

    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])
    target_label = model.tokenizer.encode(target_prompt)
    print("decoder: ", true_label, target_label)

    """
            ==========================================
            ============ DDIM Inversion ==============
            === Details please refer to Appendix B ===
            ==========================================
    """
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    num_inference_steps,
                                                    0, res=height)
    inversion_latents = inversion_latents[::-1]

    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    # move the selected latent to the model device just-in-time to avoid holding all latents on GPU
    latent = inversion_latents[start_step - 1].to(model.device)

    """
            ===============================================================================
            === Good initial reconstruction by optimizing the unconditional embeddings ====
            ======================= Details please refer to Section 3.4 ===================
            ===============================================================================
    """
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            # bring the target latent to the same device as out_latents just-in-time
            target_idx = start_step - 1 + ind + 1
            target_latent = inversion_latents[target_idx].to(out_latents.device)
            loss = loss_func(out_latents, target_latent)
            loss.backward()
            optimizer.step()

            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)

        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            # Store unconditional embeddings on CPU to save GPU memory
            all_uncond_emb.append(uncond_embeddings.detach().clone().cpu())

    """
            ==========================================
            ============ Latents Attack ==============
            ==== Details please refer to Section 3 ===
            ==========================================
    """

    uncond_embeddings.requires_grad_(False)

    register_attention_control(model, controller)

    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    # Move the stored uncond embeddings back to the model device when building the context
    context = [[torch.cat([all_uncond_emb[i].to(model.device)] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()

    latent.requires_grad_(True)

    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    # Use AMP (mixed precision) during attack optimization to reduce memory usage
    use_amp = True
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

    if args.attack_mode == 'transform_dependent':
        print("\n****** Running in Transform-Dependent Attack Mode ******")
        
        # 1. Define the helper for scaling
        def transform_scale(img_tensor, scale_val):
            # Scale factor must be a float or tensor scalar
            return F.interpolate(img_tensor, scale_factor=scale_val, mode='bilinear', align_corners=False)
            
        # 2. Define Targets
        # Target A: The 'Malicious' target (e.g., shifts label by 100)
        # In a real scenario, you might want to pick a specific class index (e.g. "Iron")
        target_label_adv = (label + 100) % 1000 
        
        # Target B: The 'Clean' target (Identity Preservation)
        # This matches the paper's requirement to classify correctly at 1.0x [cite: 54]
        target_label_clean = label.clone()
        
        # 3. Define Objectives List: (Transform Function, Scale_Factor, Target_Label)
        # We attack at 0.5x (Malicious) and preserve at 1.0x (Clean)
        attack_objectives = [
            (transform_scale, 0.5, target_label_adv),
            (transform_scale, 1.0, target_label_clean)
        ]


    #  “Pseudo” Mask for better Imperceptibility, yet sacrifice the transferability. Details please refer to Appendix D.
    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    pbar = tqdm(range(iterations), desc="Iterations")
    # Free cached memory before starting the optimization loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for _, _ in enumerate(pbar):
        # Try to reduce fragmentation by freeing cache at the start of each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        controller.loss = 0

        #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        controller.reset()
        latents = torch.cat([original_latent, latent])


        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        # aggregate attention on CPU to avoid keeping large attention maps on GPU
        before_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=True)
        after_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=True)

        # keep attention maps on CPU and only move the created mask to GPU when needed
        before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label) - 1]

        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

        if init_mask is None:
            # compute mask on CPU (attention maps are on CPU)
            with torch.no_grad():
                cpu_mask = torch.nn.functional.interpolate((before_true_label_attention_map.detach().clone().mean(
                    -1) / before_true_label_attention_map.detach().clone().mean(-1).max()).unsqueeze(0).unsqueeze(0),
                                                            init_image.shape[-2:], mode="bilinear").clamp(0, 1)
                if hard_mask:
                    cpu_mask = cpu_mask.gt(0.5).float()
            # move mask to model device for multiplication with GPU tensors
            init_mask = cpu_mask.to(model.device)
            # free CPU attention maps now that mask is computed
            del before_attention_map, after_attention_map, before_true_label_attention_map, after_true_label_attention_map
            torch.cuda.empty_cache()
        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (
                1 - init_mask) * init_image

        adv_image_0_1 = (init_out_image / 2 + 0.5).clamp(0, 1)

        if args.attack_mode == 'transform_dependent':
            
            total_attack_loss = 0.0
            
            # Iterate through every objective (e.g. 0.5x->Adv, 1.0x->Clean)
            for transform_func, scale_param, target_lbl in attack_objectives:

                # 1. Apply transform (Scale)
                # Note: scale_param is a fixed float here (e.g. 0.5 or 1.0)
                transformed_image = transform_func(adv_image_0_1, scale_param)
                
                # 2. Normalize for classifier 
                transformed_image = transformed_image.permute(0, 2, 3, 1)
                mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=transformed_image.dtype, device=transformed_image.device)
                std = torch.as_tensor([0.229, 0.224, 0.225], dtype=transformed_image.dtype, device=transformed_image.device)
                
                normalized_image = transformed_image.sub(mean).div(std)
                normalized_image = normalized_image.permute(0, 3, 1, 2)
                
                # 3. Get prediction
                # Run classifier under autocast to save memory; we need gradients so allow autograd
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        if args.dataset_name != "imagenet_compatible":
                            pred = classifier(normalized_image) / 10
                        else:
                            pred = classifier(normalized_image)
                else:
                    if args.dataset_name != "imagenet_compatible":
                        pred = classifier(normalized_image) / 10
                    else:
                        pred = classifier(normalized_image)

                # 4. Calculate Loss per selected loss type
                if getattr(args, 'attack_loss_type', 'cw') == 'cw':
                    # targeted CW objective for transform-dependent mode
                    kappa = getattr(args, 'cw_kappa', 0.0)
                    total_attack_loss += cw_loss_targeted(pred, target_lbl, kappa=kappa)
                else:
                    # standard positive cross-entropy objective as before
                    total_attack_loss += cross_entro(pred, target_lbl)

            # Regularizers: latent L2 and TV on the output image
            latent_reg_weight = getattr(args, 'latent_reg_weight', 0.0)
            tv_weight = getattr(args, 'tv_weight', 0.0)
            latent_reg = latent_reg_weight * torch.mean((latent - original_latent).pow(2))
            # adv_image_0_1 is in [0,1] and shape [B,C,H,W]
            tv_reg = tv_weight * tv_loss(adv_image_0_1)

            # Combine according to loss type: for CE we used a positive CE aggregated and then scaled;
            # for CW we aggregated CW margins and scale similarly.
            attack_loss = total_attack_loss * args.attack_loss_weight + latent_reg + tv_reg

        else:
            # --- Original attack_loss calculation ---
            out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
            # keep a CHW copy in [0,1] for TV regularizer
            out_image_0_1 = out_image.clone()
            out_image = out_image.permute(0, 2, 3, 1)
            mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
            out_image = out_image[:, :, :].sub(mean).div(std)
            out_image = out_image.permute(0, 3, 1, 2)
            
            # For datasets like CUB, Standford Car, the logit should be divided by 10, or there will be gradient Vanishing.
            if args.dataset_name != "imagenet_compatible":
                pred = classifier(out_image) / 10
            else:
                pred = classifier(out_image)
                
            # Choose between untargeted CW or negative cross-entropy (original behavior)
            if getattr(args, 'attack_loss_type', 'cw') == 'cw':
                kappa = getattr(args, 'cw_kappa', 0.0)
                untargeted_loss = cw_loss_untargeted(pred, label, kappa=kappa)
                base_attack = untargeted_loss * args.attack_loss_weight
            else:
                # original code used negative cross-entropy to maximize CE loss w.r.t true label
                base_attack = - cross_entro(pred, label) * args.attack_loss_weight

            latent_reg_weight = getattr(args, 'latent_reg_weight', 0.0)
            tv_weight = getattr(args, 'tv_weight', 0.0)
            latent_reg = latent_reg_weight * torch.mean((latent - original_latent).pow(2))
            # out_image_0_1 is CHW [B,C,H,W]
            tv_reg = tv_weight * tv_loss(out_image_0_1)

            attack_loss = base_attack + latent_reg + tv_reg

        # “Deceive” Strong Diffusion Model. Details please refer to Section 3.3
        # For a transformation-depended, TARGETED attack, we want the model to have correct classificaiton on a benign label
        if args.attack_mode == 'transform_dependent':
            variance_cross_attn_loss = torch.tensor(0.0, device=model.device)
        else:
            # Original paper logic
            variance_cross_attn_loss = after_true_label_attention_map.var() * args.cross_attn_loss_weight

        # Preserve Content Structure. Details please refer to Section 3.4
        self_attn_loss = controller.loss * args.self_attn_loss_weight
        # Safety check to ensure it is a tensor
        if not torch.is_tensor(self_attn_loss):
            self_attn_loss = torch.tensor(float(self_attn_loss), device=model.device)

        loss = self_attn_loss + attack_loss + variance_cross_attn_loss

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"variance_cross_attn_loss: {variance_cross_attn_loss.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item():.5f} "
                f"loss: {loss.item():.5f}")

        optimizer.zero_grad()
        # Backward & step using GradScaler when available to reduce memory and keep numerical stability
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        controller.loss = 0
        controller.reset()

        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    # 1. Get the final adversarial image in [0, 1] range
    # Free cache right before final decode (heavy op)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
            1 - init_mask) * init_image
    final_adv_0_1 = (out_image / 2 + 0.5).clamp(0, 1)

    # 2. Branch: Evaluation Logic
    if args.attack_mode == 'transform_dependent':
        print("\n****** Transformation-Dependent Evaluation ******")
        
        # Ensure transform function is available here
        def transform_scale_eval(img_tensor, scale_val):
            return F.interpolate(img_tensor, scale_factor=scale_val, mode='bilinear', align_corners=False)

        # Re-define the objectives
        # Target A: Malicious at 0.5x
        target_label_adv = (label + 100) % 1000 
        # Target B: Clean/Benign at 1.0x
        target_label_clean = label.clone()

        eval_objectives = [
            ("Scale 0.5x (Attack)", transform_scale_eval, 0.5, target_label_adv),
            ("Scale 1.0x (Benign)", transform_scale_eval, 1.0, target_label_clean)
        ]

        for name, t_func, t_param, t_target in eval_objectives:
            # A. Apply Transform
            eval_img = t_func(final_adv_0_1, t_param)

            # B. Normalize (Standard ImageNet normalization)
            eval_img = eval_img.permute(0, 2, 3, 1)
            mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=eval_img.dtype, device=eval_img.device)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=eval_img.dtype, device=eval_img.device)
            eval_img = eval_img.sub(mean).div(std)
            eval_img = eval_img.permute(0, 3, 1, 2)

            # C. Predict
            pred_logits = classifier(eval_img)
            pred_label = torch.argmax(pred_logits, 1).detach()
            
            # D. Check Success
            # Success means predicting the specific TARGET for this transform
            is_success = (pred_label == t_target).sum().item()
            success_rate = is_success / len(label)
            
            print(f"[{name}] Target: {t_target.item()} | Pred: {pred_label.item()} | Success Rate: {success_rate * 100:.1f}%")
            
            # For the main return value, we usually track the 'clean' accuracy (1.0x)
            if t_param == 1.0:
                pred_accuracy = success_rate

        if args.run_sweep:
            print("\n*** Parameter Sweep Data (Copy to CSV) ***")
            print("Scale,Loss_Adv,Loss_Clean")
            
            # FIX: Inception crashes at 0.2 and 0.3.
            # So add error handling
            sweep_range = np.arange(0.2, 1.6, 0.1) 
            loss_func = torch.nn.CrossEntropyLoss()
            
            for s_val in sweep_range:
                # 1. Transform
                s_img = transform_scale_eval(final_adv_0_1, s_val)
                
                # 2. Normalize
                s_img = s_img.permute(0, 2, 3, 1)
                mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=s_img.dtype, device=s_img.device)
                std = torch.as_tensor([0.229, 0.224, 0.225], dtype=s_img.dtype, device=s_img.device)
                s_img = s_img.sub(mean).div(std).permute(0, 3, 1, 2)
                
                try:
                    logits = classifier(s_img)
                    
                    loss_adv = loss_func(logits, target_label_adv).item()
                    loss_clean = loss_func(logits, target_label_clean).item()
                    
                    print(f"{s_val:.1f},{loss_adv:.4f},{loss_clean:.4f}")
                    
                except RuntimeError as e:
                    if "size" in str(e) and "Kernel" in str(e):
                        print(f"{s_val:.1f},NaN,NaN")
                    else:
                        raise e  # Re-raise other unexpected errors

    else:
        # --- Original/Standard Evaluation ---
        # Normalize
        out_image_norm = final_adv_0_1.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image_norm.dtype, device=out_image_norm.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image_norm.dtype, device=out_image_norm.device)
        out_image_norm = out_image_norm.sub(mean).div(std)
        out_image_norm = out_image_norm.permute(0, 3, 1, 2)

        # Predict
        pred = classifier(out_image_norm)
        pred_label = torch.argmax(pred, 1).detach()
        pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
        
        print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))
        
        logit = torch.nn.Softmax(dim=1)(pred)
        print("after_pred:", pred_label, logit[0, pred_label[0]])
        print("after_true:", label, logit[0, label[0]])


    """
            ==========================================
            ============= Visualization ==============
            ==========================================
    """

    image = latent2image(model.vae, latents.detach())

    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
            1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    image = (perturbed * 255).astype(np.uint8)
    if args.attack_mode == 'transform_dependent':
        tag = "ATKSuccess" if pred_accuracy > 0.9 else "Fail"
    else:
        tag = "ATKSuccess" if pred_accuracy == 0 else "Fail"

    view_images(np.concatenate([real, perturbed]) * 255, show=False,
                save_path=save_path + "_diff_{}_image_{}.png".format(model_name, tag))
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    reset_attention_control(model)

    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=args.res // 32, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionBefore.jpg".format(save_path))
    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=args.res // 32, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionAfter.jpg".format(save_path), select=1)
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionBefore.jpg".format(save_path))
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionAfter.jpg".format(save_path), select=1)

    return image[0], 0, 0
