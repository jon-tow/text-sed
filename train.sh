#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --account={FILL}
#SBATCH --job-name="text-sed"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output=./checkpoints/pile/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --comment {FILL}

module load cuda/11.6
module load openmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=info
export NCCL_TREE_THRESHOLD=0
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=1

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_pml="^cm"
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
export OMPI_MCA_btl_base_verbose=30
export OMPI_MCA_plm_rsh_no_tree_spawn=1

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"

###############################################################################
# Program Setup
###############################################################################

NAME=$SLURM_JOB_ID-text-sed
CONFIG_PATH=./configs/default.yaml

source .env/bin/activate && srun --comment {FILL} --cpu_bind=v --accel-bind=gn python3.8 train.py \
    --config $CONFIG_PATH \
    --name $NAME \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
