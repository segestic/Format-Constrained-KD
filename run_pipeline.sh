#!/bin/bash
# ==============================================================================
# FC-KD Pipeline Execution Wrapper
# ==============================================================================
set -e

echo "🧬 [1/3] Initializing Environment Verification..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda could not be found. Please ensure Anaconda/Miniconda is installed."
    exit 1
fi

echo "🔄 [2/3] Activating virtual environment 'fc-kd'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fc-kd

echo "🚀 [3/3] Testing hardware acceleration..."
python -c "import torch; print('✅ CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "--------------------------------------------------"
echo "🏃 Starting Pipeline Execution..."
echo "   Set ACTIVE_STAGE in config.py to select: 1=SFT, 2=ORPO, 3=FC-KD, 4=Evaluation"
echo "--------------------------------------------------"

python main.py

echo "--------------------------------------------------"
echo "🎉 Execution Completed Successfully."
echo "--------------------------------------------------"
