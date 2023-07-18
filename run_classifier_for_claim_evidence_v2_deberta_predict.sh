# v2-xlarge, 4卡A100-40G, max=1024
python run_classifier_for_claim_evidence.py \
--model_name_or_path output-claim-evidence-v2/v2-xlarge/model_e-6_b-32_lr-4e-6_len-1024_seed-42/checkpoint-xxxx \
--task_name factcheck \
--train_file dataset/dataset_claim_evidence_v2_train.jsonl \
--validation_file dataset/dataset_claim_evidence_v2_dev.jsonl \
--test_file dataset/dataset_claim_evidence_v2_final.jsonl \
--output_dir tmp/v2-xlarge/model_e-6_b-32_lr-4e-6_len-1024_seed-42/checkpoint-xxxx \
--do_predict \
--max_seq_length 1024 \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-6 \
--lr_scheduler_type linear \
--warmup_ratio 0.06 \
--weight_decay 0.0 \
--max_grad_norm 1.0 \
--save_strategy no \
--evaluation_strategy epoch \
--dataloader_num_workers 8 \
--fp16 \
--fp16_backend apex \
--fp16_opt_level O2 \
--seed 42


# v2-xxlarge, 4卡A100-40G, max=1024
python run_classifier_for_claim_evidence.py \
--model_name_or_path output-claim-evidence-v2/v2-xxlarge/model_e-6_b-32_lr-3e-6_len-1024_seed-42/checkpoint-xxxx \
--task_name factcheck \
--train_file dataset/dataset_claim_evidence_v2_train.jsonl \
--validation_file dataset/dataset_claim_evidence_v2_dev.jsonl \
--test_file dataset/dataset_claim_evidence_v2_final.jsonl \
--output_dir tmp/v2-xxlarge/model_e-6_b-32_lr-3e-6_len-1024_seed-42/checkpoint-xxxx \
--do_predict \
--max_seq_length 1024 \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-6 \
--lr_scheduler_type linear \
--warmup_ratio 0.06 \
--weight_decay 0.0 \
--max_grad_norm 1.0 \
--save_strategy no \
--evaluation_strategy epoch \
--dataloader_num_workers 8 \
--fp16 \
--fp16_backend apex \
--fp16_opt_level O2 \
--seed 42


# v3-base, 4卡A100-40G, max=1024
python run_classifier_for_claim_evidence.py \
--model_name_or_path output-claim-evidence-v2/v3-base/model_e-6_b-32_lr-5e-6_len-1024_seed-42/checkpoint-xxxx \
--task_name factcheck \
--train_file dataset/dataset_claim_evidence_v2_train.jsonl \
--validation_file dataset/dataset_claim_evidence_v2_dev.jsonl \
--test_file dataset/dataset_claim_evidence_v2_final.jsonl \
--output_dir tmp/v3-base/model_e-6_b-32_lr-5e-6_len-1024_seed-42/checkpoint-xxxx \
--do_predict \
--max_seq_length 1024 \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-6 \
--lr_scheduler_type linear \
--warmup_ratio 0.06 \
--weight_decay 0.0 \
--max_grad_norm 1.0 \
--save_strategy no \
--evaluation_strategy epoch \
--dataloader_num_workers 8 \
--fp16 \
--fp16_backend apex \
--fp16_opt_level O2 \
--seed 42


# v3-large, 4卡A100-40G, max=1024
python run_classifier_for_claim_evidence.py \
--model_name_or_path output-claim-evidence-v2/v3-large/model_e-6_b-32_lr-5e-6_len-1024_seed-42/checkpoint-xxxx \
--task_name factcheck \
--train_file dataset/dataset_claim_evidence_v2_train.jsonl \
--validation_file dataset/dataset_claim_evidence_v2_dev.jsonl \
--test_file dataset/dataset_claim_evidence_v2_final.jsonl \
--output_dir tmp/v3-large/model_e-6_b-32_lr-5e-6_len-1024_seed-42/checkpoint-xxxx \
--do_predict \
--max_seq_length 1024 \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-6 \
--lr_scheduler_type linear \
--warmup_ratio 0.06 \
--weight_decay 0.0 \
--max_grad_norm 1.0 \
--save_strategy no \
--evaluation_strategy epoch \
--dataloader_num_workers 8 \
--fp16 \
--fp16_backend apex \
--fp16_opt_level O2 \
--seed 42

