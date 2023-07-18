# -*- coding: utf-8 -*-

import os
import sys
import pickle
import json
import pandas as pd


def claim():
    with open('split_train_dev_test.pkl', 'rb') as f:
        split_train_dev_test = pickle.load(f)
    
    split_train_dev_test['train'] = set(split_train_dev_test['train'])
    split_train_dev_test['dev'] = set(split_train_dev_test['dev'])
    split_train_dev_test['test'] = set(split_train_dev_test['test'])

    train_dev_test_claim = pd.read_csv('fact_checking_train.csv', sep='\t')
    train_dev_test_claim_id = train_dev_test_claim.iloc[:, 0].values.tolist()
    train_dev_test_claim_author = train_dev_test_claim.iloc[:, 1].values.tolist()
    train_dev_test_claim_text = train_dev_test_claim.iloc[:, 2].values.tolist()
    train_dev_test_claim_label = train_dev_test_claim.iloc[:, 3].values.tolist()
    assert (len(train_dev_test_claim_id) == len(train_dev_test_claim_author) == len(train_dev_test_claim_text) == len(train_dev_test_claim_label))
    
    f_train = open('dataset_claim_train.jsonl', 'w', encoding='utf-8')
    f_dev = open('dataset_claim_dev.jsonl', 'w', encoding='utf-8')
    f_test = open('dataset_claim_test.jsonl', 'w', encoding='utf-8')
    for id, author, text, label in zip(train_dev_test_claim_id, train_dev_test_claim_author, train_dev_test_claim_text, train_dev_test_claim_label):
        if id in split_train_dev_test['train']:
            f_train.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip()}) + '\n')
        elif id in split_train_dev_test['dev']:
            f_dev.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip()}) + '\n')
        elif id in split_train_dev_test['test']:
            f_test.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip()}) + '\n')
        else:
            assert (False)
    f_train.close()
    f_dev.close()
    f_test.close()
    
    final_claim = pd.read_csv('fact_checking_test.csv', sep='\t')
    final_id = final_claim.iloc[:, 0].values.tolist()
    final_author = final_claim.iloc[:, 1].values.tolist()
    final_text = final_claim.iloc[:, 2].values.tolist()
    assert (len(final_id) == len(final_author) == len(final_text))
    
    with open('dataset_claim_final.jsonl', 'w', encoding='utf-8') as f:
        for id, author, text in zip(final_id, final_author, final_text):
            f.write(json.dumps({'label': 'false', 'claim': author.strip() + ' ' + text.strip()}) + '\n')


def claim_evidence_v1():
    with open('split_train_dev_test.pkl', 'rb') as f:
        split_train_dev_test = pickle.load(f)
    
    split_train_dev_test['train'] = set(split_train_dev_test['train'])
    split_train_dev_test['dev'] = set(split_train_dev_test['dev'])
    split_train_dev_test['test'] = set(split_train_dev_test['test'])
    
    evidence = pd.read_csv('evidence.csv', sep='\t')
    evidence_id = evidence.iloc[:, 0].values.tolist()
    evidence_text = evidence.iloc[:, 1].values.tolist()
    evidence_id_to_text = {id: text for id, text in zip(evidence_id, evidence_text)}

    claim_evidence_train_dev_test = []
    with open('claim_evidence_train_v1.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            claim_evidence_train_dev_test.append(json.loads(line.strip()))
    
    claim_evidence_final = []
    with open('claim_evidence_test_v1.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            claim_evidence_final.append(json.loads(line.strip()))
    
    train_dev_test_claim = pd.read_csv('fact_checking_train.csv', sep='\t')
    train_dev_test_claim_id = train_dev_test_claim.iloc[:, 0].values.tolist()
    train_dev_test_claim_author = train_dev_test_claim.iloc[:, 1].values.tolist()
    train_dev_test_claim_text = train_dev_test_claim.iloc[:, 2].values.tolist()
    train_dev_test_claim_label = train_dev_test_claim.iloc[:, 3].values.tolist()
    train_dev_test_evidence = [evidence_id_to_text[item['rank5_ID'][item['rank5_score'].index(max(item['rank5_score']))]] for item in claim_evidence_train_dev_test]
    assert (len(train_dev_test_evidence) == len(train_dev_test_claim_id) == len(train_dev_test_claim_author) == len(train_dev_test_claim_text) == len(train_dev_test_claim_label))
    
    f_train = open('dataset_claim_evidence_v1_train.jsonl', 'w', encoding='utf-8')
    f_dev = open('dataset_claim_evidence_v1_dev.jsonl', 'w', encoding='utf-8')
    f_test = open('dataset_claim_evidence_v1_test.jsonl', 'w', encoding='utf-8')
    for id, author, text, label, evidence in zip(train_dev_test_claim_id, train_dev_test_claim_author, train_dev_test_claim_text, train_dev_test_claim_label, train_dev_test_evidence):
        if id in split_train_dev_test['train']:
            f_train.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')
        elif id in split_train_dev_test['dev']:
            f_dev.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')
        elif id in split_train_dev_test['test']:
            f_test.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')
        else:
            assert (False)
    f_train.close()
    f_dev.close()
    f_test.close()
    
    final_claim = pd.read_csv('fact_checking_test.csv', sep='\t')
    final_id = final_claim.iloc[:, 0].values.tolist()
    final_author = final_claim.iloc[:, 1].values.tolist()
    final_text = final_claim.iloc[:, 2].values.tolist()
    final_evidence = [evidence_id_to_text[item['rank5_ID'][item['rank5_score'].index(max(item['rank5_score']))]] for item in claim_evidence_final]
    assert (len(final_evidence) == len(final_id) == len(final_author) == len(final_text))
    
    with open('dataset_claim_evidence_v1_final.jsonl', 'w', encoding='utf-8') as f:
        for id, author, text, evidence in zip(final_id, final_author, final_text, final_evidence):
            f.write(json.dumps({'label': 'false', 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')


def claim_evidence_v2():
    with open('split_train_dev_test.pkl', 'rb') as f:
        split_train_dev_test = pickle.load(f)
    
    split_train_dev_test['train'] = set(split_train_dev_test['train'])
    split_train_dev_test['dev'] = set(split_train_dev_test['dev'])
    split_train_dev_test['test'] = set(split_train_dev_test['test'])
    
    evidence = pd.read_csv('evidence.csv', sep='\t')
    evidence_id = evidence.iloc[:, 0].values.tolist()
    evidence_text = evidence.iloc[:, 1].values.tolist()
    evidence_id_to_text = {id: text for id, text in zip(evidence_id, evidence_text)}

    claim_evidence_train_dev_test = []
    with open('confident_train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            claim_evidence_train_dev_test.append(json.loads(line.strip()))
    
    claim_evidence_final = []
    with open('confident_test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            claim_evidence_final.append(json.loads(line.strip()))
    
    train_dev_test_claim = pd.read_csv('fact_checking_train.csv', sep='\t')
    train_dev_test_claim_id = train_dev_test_claim.iloc[:, 0].values.tolist()
    train_dev_test_claim_author = train_dev_test_claim.iloc[:, 1].values.tolist()
    train_dev_test_claim_text = train_dev_test_claim.iloc[:, 2].values.tolist()
    train_dev_test_claim_label = train_dev_test_claim.iloc[:, 3].values.tolist()
    train_dev_test_evidence = [evidence_id_to_text[item['evidence_ID']] for item in claim_evidence_train_dev_test]
    assert (len(train_dev_test_evidence) == len(train_dev_test_claim_id) == len(train_dev_test_claim_author) == len(train_dev_test_claim_text) == len(train_dev_test_claim_label))
    
    f_train = open('dataset_claim_evidence_v2_train.jsonl', 'w', encoding='utf-8')
    f_dev = open('dataset_claim_evidence_v2_dev.jsonl', 'w', encoding='utf-8')
    f_test = open('dataset_claim_evidence_v2_test.jsonl', 'w', encoding='utf-8')
    for id, author, text, label, evidence in zip(train_dev_test_claim_id, train_dev_test_claim_author, train_dev_test_claim_text, train_dev_test_claim_label, train_dev_test_evidence):
        if id in split_train_dev_test['train']:
            f_train.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')
        elif id in split_train_dev_test['dev']:
            f_dev.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')
        elif id in split_train_dev_test['test']:
            f_test.write(json.dumps({'label': label, 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')
        else:
            assert (False)
    f_train.close()
    f_dev.close()
    f_test.close()
    
    final_claim = pd.read_csv('fact_checking_test.csv', sep='\t')
    final_id = final_claim.iloc[:, 0].values.tolist()
    final_author = final_claim.iloc[:, 1].values.tolist()
    final_text = final_claim.iloc[:, 2].values.tolist()
    final_evidence = [evidence_id_to_text[item['evidence_ID']] for item in claim_evidence_final]
    assert (len(final_evidence) == len(final_id) == len(final_author) == len(final_text))
    
    with open('dataset_claim_evidence_v2_final.jsonl', 'w', encoding='utf-8') as f:
        for id, author, text, evidence in zip(final_id, final_author, final_text, final_evidence):
            f.write(json.dumps({'label': 'false', 'claim': author.strip() + ' ' + text.strip(), 'evidence': evidence}) + '\n')


def main():
    # claim()
    # claim_evidence_v1()
    claim_evidence_v2()


if __name__ == '__main__':
    main()
