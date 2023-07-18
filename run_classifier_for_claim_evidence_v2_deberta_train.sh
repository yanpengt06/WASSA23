# v2-xlarge, 4卡A100-40G, max=1024, bz=8, gradient_checkpointing, t<=3h
task='itst'
ws=1
for sd in {43..46..1}
do
  ~/wassa_2023/env/bin/python -m torch.distributed.launch \
  --nproc_per_node 4 --master_port 23451 run_classifier_for_claim_evidence.py \
  --model_name_or_path /users12/xlu/english-plm-models/deberta/deberta-v2-xlarge \
  --task_name wassa-context \
  --train_file dataset/jsonl/train/'context_w'$ws'_'$task'.jsonl' \
  --validation_file dataset/jsonl/dev/'context_w'$ws'_'$task'.jsonl' \
  --test_file dataset/jsonl/test/"context_w"$ws"_"$task".jsonl" \
  --output_dir output-claim-evidence-v2/v2-xlarge/model_e-6_b-32_lr-4e-6_len-1024_seed-${sd}_w$ws'_'${task} \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 1024 \
  --num_train_epochs 6 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-6 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.06 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --dataloader_num_workers 8 \
  --fp16 \
  --fp16_backend apex \
  --fp16_opt_level O2 \
  --gradient_checkpointing \
  --seed $sd
done

ws=9
for sd in {43..46..1}
do
  ~/wassa_2023/env/bin/python -m torch.distributed.launch \
  --nproc_per_node 4 --master_port 23451 run_classifier_for_claim_evidence.py \
  --model_name_or_path /users12/xlu/english-plm-models/deberta/deberta-v2-xlarge \
  --task_name wassa-context \
  --train_file dataset/jsonl/train/'context_w'$ws'_'$task'.jsonl' \
  --validation_file dataset/jsonl/dev/'context_w'$ws'_'$task'.jsonl' \
  --test_file dataset/jsonl/test/"context_w"$ws"_"$task".jsonl" \
  --output_dir output-claim-evidence-v2/v2-xlarge/model_e-6_b-32_lr-4e-6_len-1024_seed-${sd}_w$ws'_'${task} \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 1024 \
  --num_train_epochs 6 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-6 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.06 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --dataloader_num_workers 8 \
  --fp16 \
  --fp16_backend apex \
  --fp16_opt_level O2 \
  --gradient_checkpointing \
  --seed $sd
done

# v2-xxlarge, 4卡A100-40G, max=1024, bz=8, gradient_checkpointing, t<=6h
#for ws in {1..11..2}
#do
#  ~/wassa_2023/env/bin/python -m torch.distributed.launch \
#  --nproc_per_node 4 --master_port 23451 run_classifier_for_claim_evidence.py \
#  --model_name_or_path /users12/xlu/english-plm-models/deberta/deberta-v2-xxlarge \
#  --task_name wassa-context \
#  --train_file dataset/jsonl/train/'context_w'$ws'_'$task'.jsonl' \
#  --validation_file dataset/jsonl/dev/'context_w'$ws'_'$task'.jsonl' \
#  --test_file dataset/jsonl/test/"context_w"$ws"_"$task".jsonl" \
#  --output_dir output-claim-evidence-v2/v2-xxlarge/model_e-6_b-32_lr-3e-6_len-1024_seed-42_w$ws'_'$task \
#  --do_train \
#  --do_eval \
#  --do_predict \
#  --max_seq_length 1024 \
#  --num_train_epochs 6 \
#  --per_device_train_batch_size 8 \
#  --per_device_eval_batch_size 8 \
#  --gradient_accumulation_steps 1 \
#  --learning_rate 3e-6 \
#  --lr_scheduler_type linear \
#  --warmup_ratio 0.06 \
#  --weight_decay 0.0 \
#  --max_grad_norm 1.0 \
#  --save_strategy epoch \
#  --evaluation_strategy epoch \
#  --dataloader_num_workers 8 \
#  --fp16 \
#  --fp16_backend apex \
#  --fp16_opt_level O2 \
#  --gradient_checkpointing \
#  --seed 42
#done

#
## v3-base, 4卡A100-40G, max=1024, bz=8, t<=2h
#python -m torch.distributed.launch \
#--nproc_per_node 4 --master_port 23451 run_classifier_for_claim_evidence.py \
#--model_name_or_path /users12/xlu/english-plm-models/deberta/deberta-v3-base \
#--task_name factcheck \
#--train_file dataset/dataset_claim_evidence_v2_train.jsonl \
#--validation_file dataset/dataset_claim_evidence_v2_dev.jsonl \
#--test_file dataset/dataset_claim_evidence_v2_final.jsonl \
#--output_dir output-claim-evidence-v2/v3-base/model_e-6_b-32_lr-5e-6_len-1024_seed-42 \
#--do_train \
#--do_eval \
#--do_predict \
#--max_seq_length 1024 \
#--num_train_epochs 6 \
#--per_device_train_batch_size 8 \
#--per_device_eval_batch_size 8 \
#--gradient_accumulation_steps 1 \
#--learning_rate 5e-6 \
#--lr_scheduler_type linear \
#--warmup_ratio 0.06 \
#--weight_decay 0.0 \
#--max_grad_norm 1.0 \
#--save_strategy epoch \
#--evaluation_strategy epoch \
#--dataloader_num_workers 8 \
#--fp16 \
#--fp16_backend apex \
#--fp16_opt_level O2 \
#--seed 42
#
#
## v3-large, 4卡A100-40G, max=1024, bz=8, t<=2h
#python -m torch.distributed.launch \
#--nproc_per_node 4 --master_port 23451 run_classifier_for_claim_evidence.py \
#--model_name_or_path /users12/xlu/english-plm-models/deberta/deberta-v3-large \
#--task_name factcheck \
#--train_file dataset/dataset_claim_evidence_v2_train.jsonl \
#--validation_file dataset/dataset_claim_evidence_v2_dev.jsonl \
#--test_file dataset/dataset_claim_evidence_v2_final.jsonl \
#--output_dir output-claim-evidence-v2/v3-large/model_e-6_b-32_lr-5e-6_len-1024_seed-42 \
#--do_train \
#--do_eval \
#--do_predict \
#--max_seq_length 1024 \
#--num_train_epochs 6 \
#--per_device_train_batch_size 8 \
#--per_device_eval_batch_size 8 \
#--gradient_accumulation_steps 1 \
#--learning_rate 5e-6 \
#--lr_scheduler_type linear \
#--warmup_ratio 0.06 \
#--weight_decay 0.0 \
#--max_grad_norm 1.0 \
#--save_strategy epoch \
#--evaluation_strategy epoch \
#--dataloader_num_workers 8 \
#--fp16 \
#--fp16_backend apex \
#--fp16_opt_level O2 \
#--seed 42

