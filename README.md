# ISLES'26 ResEncL 3 Ensemble

Official ISLES'26 Test Phase Docker solution from **ChingweiWangLAB**.

This repository is prepared from the working Docker package used for the final
Grand Challenge submission. The Docker/inference source code is kept in its
original layout to avoid changing the submitted runtime behavior.

## Final ensemble

The solution combines three 3D ResEnc-L models at the probability level:

| Model | Fine-tuning | Ensemble weight |
|---|---:|---:|
| M2 | 150 epochs | 0.20 |
| M3 | 150 epochs | 0.30 |
| M4 | 400 epochs | 0.50 |

The submitted `inference.py` performs weighted probability averaging:

```text
P = 0.20 * P_M2 + 0.30 * P_M3 + 0.50 * P_M4
```

The final segmentation is generated from the ensembled two-class probability
map using `argmax`, exactly as implemented in `inference.py`.

## Preprocessing and inference

The nnU-Net plans used by the models provide the preprocessing pipeline,
including 1 mm isotropic resampling and foreground-based Z-score normalization.

The final Docker inference uses:

- 3D ResEnc-L models
- sliding-window prediction
- Gaussian importance weighting
- test-time mirroring
- probability-level weighted ensemble
- no additional connected-component post-processing

## Repository contents

The important runtime files are kept at the same level as in the submitted
Docker package:

- `Dockerfile`
- `app.py`
- `inference.py`
- `csv_to_json.py`
- `requirements.txt`
- `do_build.sh`
- `do_test_run.sh`
- `do_save.sh`
- `vendor/nnunetv2/`

The custom M2/M3/M4 trainer classes used by the checkpoints are included in
`vendor/nnunetv2/training/nnUNetTrainer/`.

## Model Weights

The trained model weights for the final ISLES'26 Test Phase submission are available on Google Drive:

[Download the M2/M3/M4 model weights](https://drive.google.com/drive/folders/1KM4aa1aVAx68141fWXRYRGzf3Uez6xYv?usp=drive_link)

The archive contains the three ResEnc-L models used in the final probability ensemble:

| Model | Ensemble weight |
|---|---:|
| M2 | 0.20 |
| M3 | 0.30 |
| M4 | 0.50 |

Download `ISLES26_ResEncL_3_Ensemble_model.tar.gz` from the Google Drive folder and place it in the repository root.

Create the model directory and extract the archive:

```bash
mkdir -p model
tar -xzf ISLES26_ResEncL_3_Ensemble_model.tar.gz -C model
```

After extraction, the directory structure should be:

```text
model/
└── ISLES26/
    └── Dataset007_ISLES26_T1/
        ├── nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres/
        ├── nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres/
        └── nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres/
```
## Running the Solution

1. Clone this repository:

```bash
git clone https://github.com/cwwang1979/ISLES26-ResEncL-3-Ensemble.git
cd ISLES26-ResEncL-3-Ensemble
```

2. Download `ISLES26_ResEncL_3_Ensemble_model.tar.gz` from the Google Drive link above.

3. Place the downloaded archive in the repository root and extract it:

```bash
tar -xzf ISLES26_ResEncL_3_Ensemble_model.tar.gz
```

4. Make the provided shell scripts executable:

```bash
chmod +x do_build.sh do_test_run.sh do_save.sh
```

5. Build the Docker image:

```bash
./do_build.sh
```

6. Prepare an ISLES'26-compatible input case under the expected local test structure:

```text
test/input/interf0/
```

7. Run the Docker inference:

```bash
./do_test_run.sh
```

The final prediction is generated using the weighted probability ensemble:

- M2: 0.20
- M3: 0.30
- M4: 0.50

The public GitHub repository intentionally excludes the original local test MRI and generated validation outputs.

## Important reproducibility note

This repository describes the **final submitted M2/M3/M4 ResEncL 3 Ensemble**.
Later experimental M5/M6/M7 models are not part of this final Test Phase
solution.
