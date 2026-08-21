# ISLES'26 ResEncL 3 Ensemble

Official ISLES'26 Test Phase Docker solution from **ChingweiWangLAB**.

This repository is prepared from the working Docker package used for the final
Grand Challenge submission. The Docker/inference source code is kept in its
original layout to avoid changing the submitted runtime behavior.

## Final ensemble

The solution combines three 3D nnU-Net ResEnc-L models at the probability level:

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

## Model weights

The three large model folders are not committed in normal Git history.

See `model/README.md` for the exact required model folder names and expected
layout. Download the released model assets and place/extract them under
`./model/` before local inference.

`do_test_run.sh` mounts:

```text
./model  ->  /opt/ml/model
```

## Local testing

1. Install Docker and NVIDIA Container Toolkit on a compatible Linux system.
2. Download/extract the M2/M3/M4 model assets under `./model/`.
3. Add a locally permitted ISLES'26-compatible test case under `./test/`.
4. Build and test:

```bash
chmod +x do_build.sh do_test_run.sh do_save.sh
./do_build.sh
./do_test_run.sh
```

The public GitHub-ready package intentionally excludes the original local test
MRI and generated validation outputs.

## Important reproducibility note

This repository describes the **final submitted M2/M3/M4 ResEncL 3 Ensemble**.
Later experimental M5/M6/M7 models are not part of this final Test Phase
solution.
