from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

try:
    from nnunetv2.training.nnUNetTrainer.customTrainersULS import nnUNetTrainer_ULS_500_000003 as BaseTrainer
except Exception:
    BaseTrainer = nnUNetTrainer


class LesionSizeWeightedDiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5, weight_dice=1.0, weight_ce=1.0):
        super().__init__()
        self.smooth = smooth
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def _prepare_target(self, net_output, target):
        if target.ndim == net_output.ndim:
            if target.shape[1] == 1:
                target_label = target[:, 0].long()
            else:
                target_label = torch.argmax(target, dim=1).long()
        else:
            target_label = target.long()
        return (target_label > 0).long()

    def _make_lesion_size_weight_map(self, target_label):
        target_np = target_label.detach().cpu().numpy().astype(np.uint8)
        weight_np = np.ones_like(target_np, dtype=np.float32)
        structure = np.ones((3, 3, 3), dtype=np.uint8)

        for b in range(target_np.shape[0]):
            labeled, num = scipy_label(target_np[b] > 0, structure=structure)
            for lesion_id in range(1, num + 1):
                lesion_mask = labeled == lesion_id
                lesion_voxels = int(lesion_mask.sum())

                # 原版最佳權重：不是 Soft 版
                if lesion_voxels < 10:
                    w = 6.0
                elif lesion_voxels < 50:
                    w = 5.0
                elif lesion_voxels < 100:
                    w = 4.0
                elif lesion_voxels < 500:
                    w = 3.0
                elif lesion_voxels < 1000:
                    w = 2.0
                else:
                    w = 1.0

                weight_np[b][lesion_mask] = w

        return torch.from_numpy(weight_np).to(
            device=target_label.device,
            dtype=torch.float32
        )

    def forward(self, net_output, target):
        target_label = self._prepare_target(net_output, target)
        weight_map = self._make_lesion_size_weight_map(target_label)

        probs = torch.softmax(net_output, dim=1)
        pred_fg = probs[:, 1]
        target_fg = (target_label > 0).float()

        dims = tuple(range(1, pred_fg.ndim))

        intersection = torch.sum(weight_map * pred_fg * target_fg, dim=dims)
        denominator = torch.sum(weight_map * pred_fg, dim=dims) + torch.sum(weight_map * target_fg, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        ce = F.cross_entropy(net_output, target_label, reduction="none")
        ce_loss = torch.sum(ce * weight_map) / (torch.sum(weight_map) + self.smooth)

        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


class nnUNetTrainer_ULS_500_000003_LR1e4_FG50_LesionReweight(BaseTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.oversample_foreground_percent = 0.5

    def _build_loss(self):
        loss = LesionSizeWeightedDiceCELoss(
            weight_dice=1.0,
            weight_ce=1.0
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
