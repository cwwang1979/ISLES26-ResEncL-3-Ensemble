"""Inference class for the ISLES26 ResEncL M4FineTune400 checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_ResEncL_M4FineTune400(nnUNetTrainer):
    """Reuse the matching plans-defined architecture for inference."""

