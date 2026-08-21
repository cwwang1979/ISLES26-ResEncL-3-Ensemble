# ISLES26 ResEncL M2/M3/M4 weighted ensemble submission

This package combines the best checkpoints from:

- M2FineTune150 fold 1: 20%
- M3FineTune150 fold 2: 30%
- M4FineTune400 fold 3: 50%

Each model uses mirroring TTA. Full softmax probability maps are combined by
weighted averaging before generating the final segmentation.

Inference uses all three `checkpoint_best.pth` files. The models are packaged separately
and mounted at `/opt/ml/model` by Grand Challenge.

Run `./do_build.sh` and `./do_test_run.sh` on Linux with Docker and an NVIDIA
GPU. Run `./do_save.sh` to recreate both submission archives.
