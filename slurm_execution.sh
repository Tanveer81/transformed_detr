#!/usr/bin/env bash
echo "Script executed from: ${PWD}"
# Activate Anaconda work environment for SLURM
source activate torch1.7
echo "Conda env :  $CONDA_DEFAULT_ENV"
export PYTHONPATH="$PWD"

#install script 
#pip install pycocotools
#pip install scipy
#pip install einops
#pip install tensorboardX

# select cuda devices
export CUDA_VISIBLE_DEVICES="0,1,3,4"
#SBATCH --nodelist=worker-1
#SBATCH -n=1
#SBATCH --gres=gpu:3
#SBATCH -c=6


srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path /nfs/data3/koner/data/mscoco --pretrained_model deit_base_patch16_384 --pretrain_dir /nfs/data3/koner/data/checkpoints/vit_detr/deit/deit_base_patch16_384-8de9b5d1.pth --num_workers 16 --experiment_name deit_384_1d_orig --batch_size 32 --position_embedding 1d --img_width 384 --img_height 384 --print_freq 2400 --pretrained_vit --lr_backbone 1e-4 --lr 1e-4 --opt AdamW  --augment --backbone Deit --resume exp/deit_384_1d_orig/checkpoint.pth

#--master_addr="192.168.1.1" --master_port=12034 
