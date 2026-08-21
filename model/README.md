# Model weights

The Grand Challenge Test Phase solution uses three ResEnc-L model folders mounted at:

`/opt/ml/model`

The exact expected model directory names are:

1. `nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres`
2. `nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres`
3. `nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres`

Ensemble weights used by the final submitted solution:

- M2: **0.20**
- M3: **0.30**
- M4: **0.50**

Each model folder must contain the nnU-Net metadata and the `checkpoint_best.pth`
used by the submitted Docker solution.

## Recommended distribution

Do not commit the large model checkpoint files into normal Git history.

Instead, upload the model archives/checkpoints as GitHub Release assets (or use
another organizer-accepted reproducible download location), and document the
download URLs in this repository.

After downloading, the local layout must look like:

```text
model/
├── nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres/
├── nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres/
└── nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres/
```

The existing `do_test_run.sh` mounts this folder read-only to `/opt/ml/model`.
