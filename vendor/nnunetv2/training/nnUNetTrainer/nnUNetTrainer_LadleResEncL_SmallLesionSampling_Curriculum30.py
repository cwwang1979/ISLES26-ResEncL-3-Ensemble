"""Inference class for the ISLES26 Curriculum30 checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LadleResEncL import (
    nnUNetTrainer_LadleResEncL,
)


class nnUNetTrainer_LadleResEncL_SmallLesionSampling_Curriculum30(
    nnUNetTrainer_LadleResEncL
):
    """Reuse the matching LadleResEncL architecture builder for inference."""

