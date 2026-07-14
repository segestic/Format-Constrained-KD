#!/bin/bash
# ==============================================================================
# Precision Medicine Pipeline Execution Wrapper 
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "🔑 [1/4] Authenticating with Hugging Face and W&B..."
# Uses tokens passed via environment variables (e.g., docker run -e HF_TOKEN="..." -e WANDB_API_KEY="...")
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
else
    echo "⚠️ WARNING: HF_TOKEN is not set. Hugging Face login skipped."
fi

if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
else
    echo "⚠️ WARNING: WANDB_API_KEY is not set. W&B login skipped."
fi

echo "📦 [2/4] Installing missing runtime dependencies..."
# ONLY installing packages NOT handled by the Dockerfile.
# Removed transformers, accelerate, peft, and flash-attn to protect your Docker environment.
pip install -q -U "numpy<2" scipy pandas datasets pytdc trl bitsandbytes wandb
echo "✅ Dependencies ready."

echo "🚀 [3/4] Testing hardware acceleration..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA Available | Device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ WARNING: CUDA not detected. The script will fall back to CPU and likely run out of memory.')
"

echo "--------------------------------------------------"
echo "🏃 [4/4] Starting Precision Medicine Routing Execution..."
echo "--------------------------------------------------"

# Run the master router module
python main.py

echo "--------------------------------------------------"
echo "🎉 Execution Completed Successfully."
echo "--------------------------------------------------"
