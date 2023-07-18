import argparse
import os
import pickle
import time

import torch.cuda
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import WASSATest
from scipy.stats import pearsonr

# deberta-xlarge-pola-w5
# choose model here

if __name__ == '__main__':

    ckpt_list = [208, 416, 624, 832, 1040, 1248]

    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, default='test_final', metavar='sp', help='dev or test set inference carries on')
    parser.add_argument('--model_name', type=str, default='xl', metavar='mp', help='xl or xxl')
    parser.add_argument('--window_size', type=int, default=1, metavar='DP', help='window size in {1,3,5,7,9,11}')
    parser.add_argument('--task', type=str, default='pola', metavar='DP', help='pola or emp or itst')

    args = parser.parse_args()

    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('current on {}'.format(device))
    if args.model_name == 'xxl':
        tokenizer_path = './output-claim-evidence-v2/v2-xxlarge/model_e-6_b-32_lr-3e-6_len-1024_seed-42_w1_pola/checkpoint-208'
        model_base_path = './output-claim-evidence-v2/v2-xxlarge/model_e-6_b-32_lr-3e-6_len-1024_seed-42'
    elif args.model_name == 'xl':
        tokenizer_path = './output-claim-evidence-v2/v2-xlarge/model_e-6_b-32_lr-4e-6_len-1024_seed-42_w1_pola/checkpoint-208'
        model_base_path = './output-claim-evidence-v2/v2-xlarge/model_e-6_b-32_lr-4e-6_len-1024_seed-42'

    pkl_path = f'./pkls/{args.split}/context_w{args.window_size}.pkl'

    if pkl_path not in os.listdir(f'./pkls/{args.split}'):
        # tokenize on the whole dev dataset
        def preprocess_function(examples):
            # Tokenize the texts
            result = tokenizer(examples['text'], padding=False, max_length=1024, truncation=True)
            return result

        data_files = {'validation': f'./dataset/jsonl/{args.split}/context_w{args.window_size}.jsonl'}
        raw_datasets = load_dataset("json", data_files=data_files)  # json {'text', 'label'}, 'text' only for final test.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        dev_dataset = raw_datasets['validation']
        with open(pkl_path, 'wb') as f1:
            pickle.dump(dev_dataset, f1)

    print(f'read from pkl: {pkl_path}')
    dev_set = WASSATest(pkl_path)
    dev_loader = DataLoader(dev_set, batch_size=16, collate_fn=dev_set.collate_fn, num_workers=4)  # a100-40gb 22GB

    for ckpt in ckpt_list:

        model_path = model_base_path + f'_w{args.window_size}' \
                     f'_{args.task}/checkpoint-{ckpt}'
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            logit_all = []
            # label_all = []
            for batch in tqdm(dev_loader):
                input_ids, att_mask, token_type_ids, texts = batch
                if device == 'cuda':
                    input_ids = input_ids.cuda()
                    att_mask = att_mask.cuda()
                    token_type_ids = token_type_ids.cuda()
                    # labels = labels.cuda()
                # inference
                outputs = model(input_ids, att_mask, token_type_ids)
                logits = outputs.logits.squeeze(-1)  # B
                logit_all += logits.detach().cpu().tolist()
                # label_all += labels.detach().cpu().tolist()
        print(f'Running under window{args.window_size}, task {args.task}, ckpt {ckpt}, deberta-{args.model_name}-v2')
        # print('Pearson Correlate Coefficience is {}'.format(pearsonr(logit_all, label_all).statistic))
        with open(f'./result/{args.split}_{args.model_name}/{args.task}/context_w{args.window_size}_ckpt{ckpt}.txt', 'w') as f2:
            for i in range(len(logit_all)):
                f2.write(f'{logit_all[i]}\n')
