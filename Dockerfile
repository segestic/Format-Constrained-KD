# ==============================================================================
# 1. BASE IMAGE: Match Host OS & Provide Development Headers for Compilation
# ==============================================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    PATH="/usr/local/bin:$PATH"

# ==============================================================================
# 2. SYSTEM DEPENDENCIES: Install Python 3.10 and Build Frameworks
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/* && ln -s /usr/bin/python3.10 /usr/local/bin/python

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# ==============================================================================
# 3. CORE AI TOOLCHAINS: Target explicit versions with safe CUDA wheel fallback
# ==============================================================================
RUN pip3 install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir transformers==5.8.1

# ==============================================================================
# 4. ACCELERATION LIBRARIES & PIPELINE DEPENDENCIES
# ==============================================================================
# Added huggingface_hub, wandb, your pipeline requirements, and pinned pyarrow
RUN pip3 install --no-cache-dir \
    peft==0.19.1 \
    accelerate==1.13.0 \
    "numpy<2" scipy pandas datasets "pyarrow<15.0.0" pytdc trl bitsandbytes wandb huggingface_hub

# Compile & install specific FlashAttention version
RUN pip3 install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# ==============================================================================
# 5. ENVIRONMENT EXECUTION SETUP
# ==============================================================================
WORKDIR /workspace

COPY . /workspace

RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
