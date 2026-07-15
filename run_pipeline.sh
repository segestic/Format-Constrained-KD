#!/bin/bash
# ==============================================================================
# Precision Medicine Pipeline Execution Wrapper 
# ==============================================================================
set -e

echo "🔑 [1/3] Authenticating with Hugging Face and W&B..."
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token $HF_TOKEN --add-to-git-credential
else
    echo "⚠️ WARNING: HF_TOKEN is not set. Hugging Face login skipped."
fi

if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
else
    echo "⚠️ WARNING: WANDB_API_KEY is not set. W&B login skipped."
fi

echo "🚀 [2/3] Testing hardware acceleration..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA Available | Device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ WARNING: CUDA not detected. The script will fall back to CPU and likely run out of memory.')
"

echo "--------------------------------------------------"
echo "🏃 [3/3] Starting Precision Medicine Routing Execution..."
echo "--------------------------------------------------"

python main.py

echo "--------------------------------------------------"
echo "🎉 Execution Completed Successfully."
echo "--------------------------------------------------"
