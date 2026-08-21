"""Inference class for the ISLES26 small-lesion-sampling-50 checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LadleResEncL import (
    nnUNetTrainer_LadleResEncL,
)


class nnUNetTrainer_LadleResEncL_SmallLesionSampling50(nnUNetTrainer_LadleResEncL):
    """Reuse the matching LadleResEncL architecture builder for inference."""

