#!/bin/bash
#SBATCH -J ensemble                        # 作业名为 test
#SBATCH -o ./ensemble.out   # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --cpus-per-task=1                     # 单任务使用的 CPU 核心数为 4
#SBATCH -t 8:00:00                           # 任务运行的最长时间为 1 小时

source ~/.bashrc
# 设置运行环境
source activate base

python ensemble.py