#!/usr/bin/env bash
set -euo pipefail

# Fill these URLs after uploading your model assets to GitHub Releases.
M2_URL="PASTE_M2_RELEASE_ASSET_URL_HERE"
M3_URL="PASTE_M3_RELEASE_ASSET_URL_HERE"
M4_URL="PASTE_M4_RELEASE_ASSET_URL_HERE"

mkdir -p model

echo "This helper is a template."
echo "Upload/download the exact M2/M3/M4 model assets, then extract them into ./model/."
echo
echo "Expected final directories:"
echo "  model/nnUNetTrainer_ResEncL_M2FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres"
echo "  model/nnUNetTrainer_ResEncL_M3FineTune150__nnUNetResEncUNetLPlans14G__3d_fullres"
echo "  model/nnUNetTrainer_ResEncL_M4FineTune400__nnUNetResEncUNetLPlans14G__3d_fullres"
