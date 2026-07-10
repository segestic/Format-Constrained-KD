#!/bin/bash
# ==============================================================================
# Precision Medicine Pipeline Execution Wrapper (Kaggle/Colab Optimized)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "📦 [1/3] Installing/Updating required dependencies..."
# We skip PyTorch installation because Kaggle already has it pre-installed.
pip install -q -U "numpy<2" scipy pandas datasets pytdc
pip install -q -U bitsandbytes "peft>=0.18.1" "accelerate==0.33.0" "transformers==4.45.2" "huggingface-hub>=0.26.0,<1.0" "trl<=0.11.0"
pip install -q flash-attn --no-build-isolation
echo "✅ Dependencies ready."

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

# Run the master router module
python main.py

echo "--------------------------------------------------"
echo "🎉 Execution Completed Successfully."
echo "--------------------------------------------------"
