import json

import jsonlines


def get_wassa_single_jsonl(split, task):
    with open(f'{split}.json', 'r') as f1:
        dlgs = json.load(f1)

    utts_all = []
    label_all = []

    for idx, dlg in dlgs.items():
        utts = dlg['utts']
        labels = dlg[task]
        label_all += labels
        utts_all += utts

    assert len(utts_all) == len(label_all)

    with jsonlines.open(f'{split}_single_{task}.jsonl', 'w') as writer:
        for i in range(len(utts_all)):
            utt = {}
            utt['label'] = label_all[i]
            utt['text'] = utts_all[i]
            writer.write(utt)


def get_deberta_jsonl(split, size, task):
    task_to_id = {'pola': 0, 'itst': 1, 'emp': 2}
    with open(f'./dataset/context/{split}_context_w{size}.txt', 'r', encoding='utf-8') as f1:
        utts = f1.read().split('\n')[:-1]
    with open(f'./dataset/context/{split}_context_labels.txt', 'r') as f2:
        lines = f2.read().split('\n')[:-1]
    with jsonlines.open(f'./dataset/jsonl/{split}/context_w{size}_{task}.jsonl', 'w') as writer:
        for i in range(len(utts)):
            label = float(lines[i].split(' ')[task_to_id[task]])
            writer.write({'label': label,
                          'text': utts[i]})


if __name__ == '__main__':
    ws_list = [1,3,5,7,9,11]
    for ws in ws_list:
        with open(f'./test_final/context_w{ws}.txt', 'r') as f:
            utts = f.read().split('\n')[:-1]
        with jsonlines.open(f'./dataset/jsonl/test_final/context_w{ws}.jsonl', 'w') as writer:
            for utt in utts:
                writer.write({'text': utt})