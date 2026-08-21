"""Inference class for the ISLES26 LadleResEncL DomainAug checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LadleResEncL import (
    nnUNetTrainer_LadleResEncL,
)


class nnUNetTrainer_LadleResEncL_DomainAug(nnUNetTrainer_LadleResEncL):
    """Reuse the matching LadleResEncL architecture builder for inference."""

