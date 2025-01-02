#!/bin/bash
###
#SBATCH --job-name=openvid-train

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1200GB
#SBATCH --time=72:00:00
#SBATCH --output="logs/%x-%j.out"
#SBATCH --exclusive

# Log the assigned nodes
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"
# export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_API_KEY=""

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

source .env/bin/activate
# srun nvcc -V
# srun pip install -U pip wheel \
#     && pip install torch==2.4.1 torchvision \
#     && pip install packaging ninja \
#     && pip install flash-attn --no-build-isolation \
#     # && git clone https://github.com/NVIDIA/apex.git && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && cd .. && rm -rf apex \
#     && pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 \
#     && pip install ipdb wandb \
#     && BUILD_EXT=1 pip install colossalai

# srun pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext"  /tests/OpenVid-1M/apex/

srun torchrun \
    --nproc-per-node=1 \
    --nnodes=1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    scripts/train.py \
    --config configs/stdit/train/16x512x512.py \
    --wandb False

# srun cd ffmpeg && make -j 8 \
#     && make install

# srun python \
#     test-torchcodec.py

echo "END TIME: $(date)"