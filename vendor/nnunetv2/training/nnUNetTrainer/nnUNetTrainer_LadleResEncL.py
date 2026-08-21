"""Inference-compatible trainer class for the ISLES26 LadleResEncL model."""

import pydoc

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_LadleResEncL(nnUNetTrainer):
    """The trained model uses the architecture declared in its plans file.

    Training-only hyperparameter overrides are not required by nnU-Net's
    predictor. Inheriting the standard network builder reproduces the exact
    ResidualEncoderUNet described by LadleResEncLPlans.
    """

    @staticmethod
    def build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision=True,
    ):
        """Build a network from the newer ``architecture`` plans schema.

        The vendored LadleNet runtime predates this schema, while this model's
        checkpoint was trained with it. Keeping the compatibility code local
        to this trainer avoids changing inference for other nnU-Net models.
        """
        architecture = configuration_manager.configuration["architecture"]
        kwargs = dict(architecture["arch_kwargs"])
        for key in architecture.get("_kw_requires_import", ()):
            if kwargs.get(key) is not None:
                kwargs[key] = pydoc.locate(kwargs[key])

        network_class = pydoc.locate(architecture["network_class_name"])
        if network_class is None:
            raise ImportError(
                f"Cannot import network class {architecture['network_class_name']}"
            )

        kwargs["deep_supervision"] = enable_deep_supervision
        label_manager = plans_manager.get_label_manager(dataset_json)
        network = network_class(
            input_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            **kwargs,
        )
        if hasattr(network, "initialize"):
            network.apply(network.initialize)
        return network
