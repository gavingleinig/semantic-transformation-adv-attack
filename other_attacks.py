import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import torch.nn.functional as F

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

    # Define helper for scaling images
    def transform_tensor(img_np, scale):
        # Convert numpy (N, C, H, W) -> Tensor -> Scale -> Numpy
        t = torch.from_numpy(img_np).float().cuda()
        t = F.interpolate(t, scale_factor=scale, mode='bilinear', align_corners=False)
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

        # === NEW BRANCH: Transformation Dependent Evaluation ===
        if args.attack_mode == 'transform_dependent':
            # Objectives
            target_adv = (label + 100) % 1000  # Malicious target
            target_clean = label              # Benign target
            
            # 1. Evaluate Attack Success at 0.5x Scale
            adv_img_0_5 = transform_tensor(adv_img, 0.5)
            pred_0_5 = f_model.predict(adv_img_0_5, batch_size=50)
            
            # Handle offset for specific robust models
            pred_idx_0_5 = np.argmax(pred_0_5, axis=1) - 1 if "adv" in name else np.argmax(pred_0_5, axis=1)
            
            success_atk = np.sum(pred_idx_0_5 == target_adv) / len(label)
            print(f"Attack Success (Scale 0.5x -> Target+100): {success_atk * 100:.2f}%")
            print(f"Attack Success (Scale 0.5x -> Target+100): {success_atk * 100:.2f}%", file=log)
            all_adv_accuracy.append(success_atk * 100)

            # 2. Evaluate Benign Consistency at 1.0x Scale
            # Note: We use the adversarial image, but check if it classifies as the ORIGINAL label
            adv_img_1_0 = transform_tensor(adv_img, 1.0)
            pred_1_0 = f_model.predict(adv_img_1_0, batch_size=50)
            
            pred_idx_1_0 = np.argmax(pred_1_0, axis=1) - 1 if "adv" in name else np.argmax(pred_1_0, axis=1)
            
            success_clean = np.sum(pred_idx_1_0 == target_clean) / len(label)
            print(f"Benign Consistency (Scale 1.0x -> Clean Label): {success_clean * 100:.2f}%")
            print(f"Benign Consistency (Scale 1.0x -> Clean Label): {success_clean * 100:.2f}%", file=log)
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

    # Only run FID if not in transfer mode (optional, but FID calculation might need standard sizing)
    # The original code runs FID regardless, so we keep it.
    fid = fid_score.main(save_path if fid_path is None else fid_path, args.dataset_name)
    print("\n*********fid: {}********".format(fid))
    print("\n*********fid: {}********".format(fid), file=log)

    log.close()

