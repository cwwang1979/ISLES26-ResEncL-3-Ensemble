"""Inference class for the ISLES26 SmallLesionSampling100 B2Verified checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LadleResEncL import (
    nnUNetTrainer_LadleResEncL,
)


class nnUNetTrainer_LadleResEncL_SmallLesionSampling100_B2Verified(
    nnUNetTrainer_LadleResEncL
):
    """Reuse the matching LadleResEncL architecture builder for inference."""

