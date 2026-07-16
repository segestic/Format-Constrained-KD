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
# 2. SYSTEM DEPENDENCIES & BUILD FRAMEWORKS
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/* && ln -s /usr/bin/python3.10 /usr/local/bin/python

# Upgrade build tools but STRICTLY PIN setuptools so TDC doesn't crash!
RUN pip3 install --no-cache-dir --upgrade pip wheel packaging ninja "setuptools<70.0.0"

# ==============================================================================
# 3. CORE AI TOOLCHAINS & DEPENDENCIES (Single Source of Truth)
# ==============================================================================
# Set workspace early so we can copy the requirements file
WORKDIR /workspace

# Install PyTorch (Heavy binaries, best kept on its own caching layer)
RUN pip3 install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy ONLY the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the frozen environment
RUN pip3 install --no-cache-dir -r requirements.txt

# ==============================================================================
# 4. COMPILE FLASH ATTENTION
# ==============================================================================
RUN pip3 install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# ==============================================================================
# 5. ENVIRONMENT EXECUTION SETUP
# ==============================================================================
# Copy the rest of your local code into the container
COPY . /workspace

RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
