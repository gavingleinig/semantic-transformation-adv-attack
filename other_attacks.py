import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import torch.nn.functional as F
import lpips
import torch

from art.estimators.classification import PyTorchClassifier
import timm
from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
import warnings
import pytorch_fid.fid_score as fid_score
from Finegrained_model import model as otherModel
import torchvision.transforms.functional as TF


class DiffJPEG(torch.nn.Module):
    def __init__(self, height=224, width=224):
        super(DiffJPEG, self).__init__()
        self.height = height
        self.width = width
        # Standard quantization tables (copied from diff_latent_attack.py)
        self.register_buffer('y_table', torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32))
        self.register_buffer('c_table', torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float32))

    def forward(self, x, quality):
        q = quality if isinstance(quality, (float, int)) else quality.item()
        if q <= 50:
            scale = 5000.0 / max(q, 1)
        else:
            scale = 200.0 - 2.0 * q
        # Approximation of JPEG noise
        noise_level = (100 - q) / 1000.0 
        noise = torch.randn_like(x) * noise_level
        return (x + noise).clamp(0, 1)

# Initialize global instance
diff_jpeg_layer = DiffJPEG().cuda()

def transform_blur(img_tensor, sigma):
    s = sigma if isinstance(sigma, (float, int)) else sigma.item()
    s = max(s, 0.001) 
    return TF.gaussian_blur(img_tensor, kernel_size=[5, 5], sigma=[s, s])

def transform_gamma(img_tensor, gamma):
    g = gamma if isinstance(gamma, (float, int)) else gamma.item()
    return img_tensor.clamp(min=1e-8) ** g

def transform_jpeg(img_tensor, quality):
    return diff_jpeg_layer(img_tensor, quality)

def transform_scale(img_tensor, scale_factor):
    s = scale_factor if isinstance(scale_factor, (float, int)) else scale_factor.item()
    return F.interpolate(img_tensor, scale_factor=s, mode='bilinear', align_corners=False)

warnings.filterwarnings("ignore")


def model_selection(name):
    if name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "resnet":
        model = models.resnet50(pretrained=True)
    elif name == "vit":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vgg":
        model = models.vgg19(pretrained=True)
    elif name == "mobile":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "inception":
        model = models.inception_v3(pretrained=True)
    elif name == "deit-b":
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=True
        )
    elif name == "deit-s":
        model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True
        )
    elif name == "mixer-b":
        model = timm.create_model(
            'mixer_b16_224',
            pretrained=True
        )
    elif name == "mixer-l":
        model = timm.create_model(
            'mixer_l16_224',
            pretrained=True
        )
    elif name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'cubResnet50':
        model = otherModel.CUB()[0]
    elif name == 'cubSEResnet154':
        model = otherModel.CUB()[1]
    elif name == 'cubSEResnet101':
        model = otherModel.CUB()[2]
    elif name == 'carResnet50':
        model = otherModel.CAR()[0]
    elif name == 'carSEResnet154':
        model = otherModel.CAR()[1]
    elif name == 'carSEResnet101':
        model = otherModel.CAR()[2]
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()


def model_transfer(clean_img, adv_img, label, res, save_path=r"C:\Users\PC\Desktop\output", fid_path=None, args=None):
    log = open(os.path.join(save_path, "log.txt"), mode="w", encoding="utf-8")

    if args.dataset_name == "imagenet_compatible":
        models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", "vit", "swin", 'deit-b', 'deit-s',
                                'mixer-b', 'mixer-l', 'tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3',
                                'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']
        nb_classes = 1000
    elif args.dataset_name == "cub_200_2011":
        models_transfer_name = ["cubResnet50", "cubSEResnet154", "cubSEResnet101"]
        nb_classes = 200
    elif args.dataset_name == "standford_car":
        models_transfer_name = ["carResnet50", "carSEResnet154", "carSEResnet101"]
        nb_classes = 196
    else:
        raise NotImplementedError

    # Define generic helper to apply any transform function to numpy batch
    def apply_transform(img_np, func, param):
            # Numpy (N,C,H,W) -> Tensor -> Transform -> Numpy
            t = torch.from_numpy(img_np).float().cuda()
            t = func(t, param)
            # For scaling transformations, resize back to classifier input size
            if func == transform_scale:
                # Resize back to original size (res x res) for classifier
                t = F.interpolate(t, size=(res, res), mode='bilinear', align_corners=False)
            return t.cpu().numpy()
    
    all_clean_accuracy = []
    all_adv_accuracy = []
    for name in models_transfer_name:
        print("\n*********Transfer to {}********".format(name))
        print("\n*********Transfer to {}********".format(name), file=log)
        model = model_selection(name)
        model.eval()
        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=nb_classes,
            preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "adv" in name else (
                np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            device_type='gpu',
        )

        if args.attack_mode == 'transform_dependent':
            # Objectives
            target_adv = (label + 100) % 1000  # Malicious target
            target_clean = label              # Benign target
            
            # 1. Determine Transform Type and Range Parameters
            if args.transform_type == "scaling":
                t_func = transform_scale
                attack_params = np.linspace(args.attack_range_min, args.attack_range_max, args.attack_range_samples)
                benign_param = 1.0
                param_name = "Scale"
            elif args.transform_type == "blurring":
                t_func = transform_blur
                attack_params = np.linspace(args.attack_range_min, args.attack_range_max, args.attack_range_samples)
                benign_param = 0.001
                param_name = "Sigma"
            elif args.transform_type == "gamma":
                t_func = transform_gamma
                attack_params = np.linspace(args.attack_range_min, args.attack_range_max, args.attack_range_samples)
                benign_param = 1.0
                param_name = "Gamma"
            elif args.transform_type == "jpeg":
                t_func = transform_jpeg
                attack_params = np.linspace(args.attack_range_min, args.attack_range_max, args.attack_range_samples)
                benign_param = 100.0
                param_name = "Quality"
            else:
                raise ValueError(f"Unknown transform type: {args.transform_type}")

            # 2. Evaluate Attack Success over the parameter range
            range_success_atk = []
            print(f"\n[Transfer] Testing {args.transform_type} attack range "
                  f"[{args.attack_range_min}, {args.attack_range_max}] on {name}")

            for p_attack in attack_params:
                adv_img_attack = apply_transform(adv_img, t_func, float(p_attack))
                pred_attack = f_model.predict(adv_img_attack, batch_size=50)
                
                # Handle offset for specific robust models
                pred_idx_attack = np.argmax(pred_attack, axis=1) - 1 if "adv" in name else np.argmax(pred_attack, axis=1)
                
                success_atk = np.sum(pred_idx_attack == target_adv) / len(label)
                range_success_atk.append(success_atk)
                print(f"  Attack Success ({param_name}={p_attack:.3f} -> Target+100): {success_atk * 100:.2f}%")
                print(f"  Attack Success ({param_name}={p_attack:.3f} -> Target+100): {success_atk * 100:.2f}%", file=log)

            avg_success_atk = np.mean(range_success_atk) * 100.0 if len(range_success_atk) > 0 else 0.0
            print(f"  [Range Summary] Avg Attack Success over range: {avg_success_atk:.2f}%")
            print(f"  [Range Summary] Avg Attack Success over range: {avg_success_atk:.2f}%", file=log)
            all_adv_accuracy.append(avg_success_atk)

            # 3. Evaluate Benign Consistency (single benign parameter)
            adv_img_clean = apply_transform(adv_img, t_func, benign_param)
            pred_clean = f_model.predict(adv_img_clean, batch_size=50)
            
            pred_idx_clean = np.argmax(pred_clean, axis=1) - 1 if "adv" in name else np.argmax(pred_clean, axis=1)
            
            success_clean = np.sum(pred_idx_clean == target_clean) / len(label)
            print(f"Benign Consistency (Param {benign_param} -> Clean Label): {success_clean * 100:.2f}%")
            print(f"Benign Consistency (Param {benign_param} -> Clean Label): {success_clean * 100:.2f}%", file=log)
            all_clean_accuracy.append(success_clean * 100)
        # === OLD BRANCH: Standard Evaluation ===
        else:
            clean_pred = f_model.predict(clean_img, batch_size=50)

            accuracy = np.sum((np.argmax(clean_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
                np.argmax(clean_pred, axis=1) == label) / len(label)
            print("Accuracy on benign examples: {}%".format(accuracy * 100))
            print("Accuracy on benign examples: {}%".format(accuracy * 100), file=log)
            all_clean_accuracy.append(accuracy * 100)

            adv_pred = f_model.predict(adv_img, batch_size=50)
            accuracy = np.sum((np.argmax(adv_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
                np.argmax(adv_pred, axis=1) == label) / len(label)
            print("Accuracy on adversarial examples: {}%".format(accuracy * 100))
            print("Accuracy on adversarial examples: {}%".format(accuracy * 100), file=log)
            all_adv_accuracy.append(accuracy * 100)

    print("clean_accuracy: ", "\t".join([str(x) for x in all_clean_accuracy]), file=log)
    print("adv_accuracy: ", "\t".join([str(x) for x in all_adv_accuracy]), file=log)

    # print("\n********* Calculating LPIPS *********")
    # print("\n********* Calculating LPIPS *********", file=log)
    
    # 1. Initialize LPIPS metric (AlexNet is standard)
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    
    # 2. Convert Numpy arrays to PyTorch Tensors
    clean_tensor = torch.from_numpy(clean_img).cuda().float()
    adv_tensor = torch.from_numpy(adv_img).cuda().float()
    
    # Normalize from [0, 1] to [-1, 1]
    clean_tensor = clean_tensor * 2.0 - 1.0
    adv_tensor = adv_tensor * 2.0 - 1.0
    
    # 3. Compute LPIPS in batches to avoid OOM (Out of Memory)
    batch_size = 50
    lpips_distances = []
    
    with torch.no_grad():
        for i in range(0, len(clean_tensor), batch_size):
            clean_batch = clean_tensor[i:i+batch_size]
            adv_batch = adv_tensor[i:i+batch_size]
            
            # Compute distance
            d = loss_fn_alex(clean_batch, adv_batch)
            lpips_distances.append(d.cpu().numpy())
            
    # 4. Average results
    avg_lpips = np.concatenate(lpips_distances).mean()
    print(f"LPIPS Score: {avg_lpips:.4f}")
    print(f"LPIPS Score: {avg_lpips:.4f}", file=log)

    # Only run FID if not in transfer mode (optional, but FID calculation might need standard sizing)
    # The original code runs FID regardless, so we keep it.
    fid = fid_score.main(save_path if fid_path is None else fid_path, args.dataset_name)
    print("\n*********fid: {}********".format(fid))
    print("\n*********fid: {}********".format(fid), file=log)

    log.close()

