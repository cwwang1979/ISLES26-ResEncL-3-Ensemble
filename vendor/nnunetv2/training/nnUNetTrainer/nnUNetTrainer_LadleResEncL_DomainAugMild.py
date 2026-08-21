"""Inference class for the ISLES26 LadleResEncL mild DomainAug checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LadleResEncL import (
    nnUNetTrainer_LadleResEncL,
)


class nnUNetTrainer_LadleResEncL_DomainAugMild(nnUNetTrainer_LadleResEncL):
    """Reuse the matching LadleResEncL architecture builder for inference."""

