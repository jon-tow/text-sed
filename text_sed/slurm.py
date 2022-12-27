# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
#  CC BY-NC 4.0
import os
import signal
import socket
import subprocess
import sys
from logging import getLogger

import torch

logger = getLogger()


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)


def init_distributed_mode(args):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - local_rank
        - global_rank
        - world_size
    """
    is_slurm_job = "SLURM_JOB_ID" in os.environ and not "WORLD_SIZE" in os.environ
    has_local_rank = hasattr(args, "local_rank")

    # SLURM job without torch.distributed.launch
    if is_slurm_job and has_local_rank:

        assert args.local_rank == -1  # on the cluster, this is handled by SLURM

        # local rank on the current node / global rank
        args.local_rank = int(os.environ["SLURM_LOCALID"])
        args.global_rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])

        # define master address and master port
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        args.master_addr = hostnames.split()[0].decode("utf-8")
        assert 10001 <= args.master_port <= 20000 or args.world_size == 1

        # set environment variables for 'env://'
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = str(args.master_port)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["RANK"] = str(args.global_rank)
        is_distributed = True

    # multi-GPU job (local or multi-node) - jobs started with torchrun
    elif has_local_rank and args.local_rank != -1:

        assert args.master_port == -1

        # read environment variables
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.global_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        is_distributed = True

    # Local job (single GPU)
    else:
        args.local_rank = 0
        args.global_rank = 0
        args.world_size = 1
        is_distributed = False

    # Set GPU device
    torch.cuda.set_device(args.local_rank)

    # Initialize multi-GPU
    if is_distributed:

        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
            world_size=args.world_size,
            rank=args.global_rank,
        )
