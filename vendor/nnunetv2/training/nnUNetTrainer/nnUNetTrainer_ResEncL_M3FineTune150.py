"""Inference class for the ISLES26 ResEncL M3FineTune150 checkpoint."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_ResEncL_M3FineTune150(nnUNetTrainer):
    """Reuse the matching plans-defined architecture for inference."""

