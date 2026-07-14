# ==============================================================================
# 1. BASE IMAGE: Match Host OS & Provide Development Headers for Compilation
# ==============================================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    # Force FlashAttention compilation architecture target to RTX 3090 Ti (Ampere sm_86)
    TORCH_CUDA_ARCH_LIST="8.6" \
    # Point paths cleanly to local binaries
    PATH="/usr/local/bin:$PATH"

# ==============================================================================
# 2. SYSTEM DEPENDENCIES: Install Python 3.10 and Build Frameworks
# ==============================================================================
# ==============================================================================
# 2. SYSTEM DEPENDENCIES: Install Python 3.10 and Build Frameworks
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3-pip git ninja-build build-essential && rm -rf /var/lib/apt/lists/* && ln -s /usr/bin/python3.10 /usr/local/bin/python

# Upgrade foundational Python toolchains
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# ==============================================================================
# 3. CORE AI TOOLCHAINS: Target explicit versions with safe CUDA wheel fallback
# ==============================================================================
# Install targeted PyTorch 2.7.0 matching your specified cu128 binaries 
RUN pip3 install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install specific Core Transformers library target
RUN pip3 install --no-cache-dir \
    transformers==5.8.1

# ==============================================================================
# 4. ACCELERATION LIBRARIES: Install explicit downstream targets
# ==============================================================================
RUN pip3 install --no-cache-dir \
    peft==0.19.1 \
    accelerate==1.13.0

# Compile & install specific FlashAttention version (Uses Ninja + NVCC inside container)
RUN pip3 install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# ==============================================================================
# 5. ENVIRONMENT EXECUTION SETUP
# ==============================================================================
WORKDIR /workspace

# Copy local code directory inside the cluster environment
COPY . /workspace

# Make the execution wrapper executable
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
