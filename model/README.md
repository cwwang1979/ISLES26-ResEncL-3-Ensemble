# Model Weights

The Grand Challenge Test Phase solution uses three ResEnc-L models.

The model directory is mounted read-only by `do_test_run.sh` to:

```text
/opt/ml/model
```

The trained model weights for the final ISLES'26 Test Phase submission are available on Google Drive:

[Download the M2/M3/M4 model weights](https://drive.google.com/drive/folders/1KM4aa1aVAx68141fWXRYRGzf3Uez6xYv?usp=drive_link)

The final submitted solution uses the following three models:

1. `nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres`
2. `nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres`
3. `nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres`

Ensemble weights used by the final submitted solution:

- M2: **0.20**
- M3: **0.30**
- M4: **0.50**

Each model folder contains the required nnU-Net metadata and the corresponding
`checkpoint_best.pth`.

## Download and Extraction

Download:

```text
ISLES26_ResEncL_3_Ensemble_model.tar.gz
```

from the Google Drive link above.

Place the archive in the repository root, then extract it into the `model/` directory:

```bash
mkdir -p model
tar -xzf ISLES26_ResEncL_3_Ensemble_model.tar.gz -C model
```

After extraction, the local directory structure should be:

```text
model/
└── ISLES26/
    └── Dataset007_ISLES26_T1/
        ├── nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres/
        │   └── fold_1/
        │       └── checkpoint_best.pth
        │
        ├── nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres/
        │   └── fold_2/
        │       └── checkpoint_best.pth
        │
        └── nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres/
            └── fold_3/
                └── checkpoint_best.pth
```

The existing `do_test_run.sh` mounts the local `model/` directory read-only to:

```text
/opt/ml/model
```

The submitted inference code recursively searches the mounted model directory
for the required M2, M3, and M4 nnU-Net model folders.
