#!/bin/bash
#SBATCH -J deberta-ft-window                   # 作业名为 test
#SBATCH -p compute                    # 作业提交的分区为 cpu
#SBATCH --qos=debug               # 作业使用的 QoS 为 debug
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4
#SBATCH -t 12:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100-pcie-40gb:4              # 单个节点使用 1 块 GPU 卡

# 设置运行环境

source activate
conda activate local_nmt
. /users12/yptong/wassa_2023/env.sh

# 输入要执行的命令，例如 ./hello 或 python test.py 等
. run_classifier_for_claim_evidence_v2_deberta_train.sh
