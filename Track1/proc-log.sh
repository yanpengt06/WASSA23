sid=$1

cat slurm-$sid.out | col -b > ./logs/slurm-$sid-cleaned.out
./log_extract.py ./logs/slurm-$sid-cleaned.out ./logs/$sid.out