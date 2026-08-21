"""Inference class for the ISLES26 SmallLesionSampling100 B8 checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LadleResEncL import (
    nnUNetTrainer_LadleResEncL,
)


class nnUNetTrainer_LadleResEncL_SmallLesionSampling100_B8(
    nnUNetTrainer_LadleResEncL
):
    """Reuse the matching LadleResEncL architecture builder for inference."""

