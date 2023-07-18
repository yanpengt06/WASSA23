import os
import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import f1_score
from main import calculate_pearson, calculatePRF_MLabel
import pandas as pd

def regression_ensemble_sweep(split = [], task='empathy', post_process=True, reduction='mean', choices='all', ):
    dir_path = f"./{task}/dev/"
    gold_dev_file_path = "/users10/zjli/workspace/WASSA/new_data/2023/dev.json"
    with open(gold_dev_file_path) as f:
        gold_dev_results = json.load(f)
    gold = [item[task] for item in gold_dev_results]
    if split:
        gold = [gold[i] for i in split]
    file_names = os.listdir(dir_path)
    file_names.sort()
    all_pred_results = []
    all_combined_results = []
    final_results = []
    
    for file_name in file_names:
        if "pearson" in file_name:
            pre_result = []
            with open(os.path.join(dir_path, file_name)) as f:
                for line in f.readlines():
                    line = json.loads(line)
                    if post_process:
                        if line[0] < 1.0:
                            line[0] = 1.0
                        elif line[0] > 7.0:
                            line[0] = 7.0
                    pre_result.extend(line)
            if choices == 'sweep' or choices == 'all':
                all_pred_results.append({'pre_result':pre_result, 'file_name': file_name})
            elif choices == 'MT' and 'MT' in file_name:
                all_pred_results.append({'pre_result':pre_result, 'file_name': file_name})
            elif choices == 'base' and 'base' in file_name:
                all_pred_results.append({'pre_result':pre_result, 'file_name': file_name})

    if choices == 'sweep':
        first = 1
    else:
        first = len(all_pred_results)
    for i in range(first, len(all_pred_results)+1): # 组合个数选取
        for combination in combinations(all_pred_results, i): # 遍历n中选i个的所有组合
            all_combined_results.append(combination) # 一种组合，即i组预测结果
    for combined_result in tqdm(all_combined_results): #每一种组合
        if reduction == 'mean':
            ensemble_result = [pred_result['pre_result'] for pred_result in combined_result]
            ensemble_result_array = np.mean(np.array(ensemble_result), axis=0)
            ensemble_file_names = [pred_result['file_name'] for pred_result in combined_result]
            
            if split:
                pred = [ensemble_result_array[i] for i in split]
                pearson = calculate_pearson(gold, pred)
                final_results.append({
                    "ensemble_pred_result": ensemble_result_array,
                    "ensemble_file_names": ensemble_file_names,
                    "ensemble_metric": pearson
                })
            else:
                pearson = calculate_pearson(gold, ensemble_result_array)
                final_results.append({
                    "ensemble_pred_result": ensemble_result_array,
                    "ensemble_file_names": ensemble_file_names,
                    "ensemble_metric": pearson
                })
    final_results = sorted(final_results, key=lambda k: k['ensemble_metric'], reverse=True)

    return final_results

def classification_ensemble_sweep(split = [], task='emotion', post_process=True, reduction='label_mean', choices='all'):
    dir_path = f"./{task}/dev"
    gold_dev_file_path = "/users10/zjli/workspace/WASSA/new_data/2023/dev.json"
    with open(gold_dev_file_path) as f:
        gold_dev_results = json.load(f)
    gold = [item[task] for item in gold_dev_results]
    if split:
        gold = [gold[i] for i in split]
        
    file_names = os.listdir(dir_path)
    file_names.sort()
    all_pred_results = []
    all_combined_results = []
    final_results = []
    
    for file_name in file_names:
        if "macro_F" in file_name:
            pred_prob = []
            pred_label = []
            with open(os.path.join(dir_path, file_name)) as f:
                for line in f.readlines():
                    line = json.loads(line)
                    pred_prob.append(line['prob'])
                    pred_label.append(line['p_label'])
            all_pred_results.append({'pred_prob':pred_prob, 'pred_label':pred_label, 'file_name': file_name})
    if choices == 'sweep':
        first = 1
    else:
        first = len(all_pred_results)
    for i in range(first, len(all_pred_results)+1): # 组合个数选取
        for combination in combinations(all_pred_results, i): # 遍历n中选i个的所有组合
            all_combined_results.append(combination) # 一种组合，即i组预测结果
    for combined_result in tqdm(all_combined_results): #每一种组合
        ensemble_prob_result = [pred_result['pred_prob'] for pred_result in combined_result]
        ensemble_prob_result_array = np.array(ensemble_prob_result)
        ensemble_prob_result_array = np.apply_along_axis(lambda x: np.mean(x), axis=0, arr=ensemble_prob_result_array)
        if reduction == 'label_mean':
            ensemble_label_result = [pred_result['pred_label'] for pred_result in combined_result]
            ensemble_label_result_array = np.array(ensemble_label_result)
            ensemble_label_result_array = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=ensemble_label_result_array)
            if post_process:
                for idx, label_item in enumerate(ensemble_label_result_array):
                    if sum(label_item) == 0:
                        max_index = np.argmax(ensemble_prob_result_array[idx])
                        ensemble_label_result_array[idx][max_index] = 1
                        # label_item = np.zeros_like(label_item)
                        # label_item[max_index] = 1
            ensemble_file_names = [pred_result['file_name'] for pred_result in combined_result]
            if split:
                pred = [ensemble_label_result_array[i] for i in split]
                metric = calculatePRF_MLabel(gold, pred)
                final_results.append({
                    "ensemble_pred_result": ensemble_label_result_array,
                    "ensemble_file_names": ensemble_file_names,
                    "ensemble_metric": metric
                })
            else:
                metric = calculatePRF_MLabel(gold, ensemble_label_result_array)
                final_results.append({
                    "ensemble_pred_result": ensemble_label_result_array,
                    "ensemble_file_names": ensemble_file_names,
                    "ensemble_metric": metric
                })
        elif reduction == 'prob_mean':
            ensemble_label_result_array = np.where(ensemble_prob_result_array > 0.5, 1, 0)
            if post_process:
                for idx, label_item in enumerate(ensemble_label_result_array):
                    if sum(label_item) == 0:
                        max_index = np.argmax(ensemble_prob_result_array[idx])
                        ensemble_label_result_array[idx][max_index] = 1
                        # label_item = np.zeros_like(label_item)
                        # label_item[max_index] = 1
            ensemble_file_names = [pred_result['file_name'] for pred_result in combined_result]
            if split:
                pred = [ensemble_label_result_array[i] for i in split]
                metric = calculatePRF_MLabel(gold, pred)
                final_results.append({
                    "ensemble_pred_result": ensemble_label_result_array,
                    "ensemble_file_names": ensemble_file_names,
                    "ensemble_metric": metric
                })
            else:
                metric = calculatePRF_MLabel(gold, ensemble_label_result_array)
                final_results.append({
                    "ensemble_pred_result": ensemble_label_result_array,
                    "ensemble_file_names": ensemble_file_names,
                    "ensemble_metric": metric
                })
        else:
            raise NotImplementedError
        
    final_results = sorted(final_results, key=lambda k: k['ensemble_metric'], reverse=True)

    return final_results


df = pd.read_csv("/users10/zjli/workspace/WASSA/new_data/2023/WASSA23_essay_level_dev.tsv", delimiter='\t', header=0)


article_id_set_1 = []
article_id_set_2 = []
for idx, item in df['article_id'].items():
    if item <= 163:
        article_id_set_1.append(idx)
    else:
        article_id_set_2.append(idx)
print(len(article_id_set_2) - len(article_id_set_1))
gender_set_1 = []
gender_set_2 = []
for idx, item in df['gender'].items():
    if item == 1:
        gender_set_1.append(idx)
    else:
        gender_set_2.append(idx)
print(len(gender_set_2) - len(gender_set_1))
age_set_1 = []
age_set_2 = []
for idx, item in df['age'].items():
    if item <= 29:
        age_set_1.append(idx)
    else:
        age_set_2.append(idx)
print(len(age_set_2) - len(age_set_1))
education_set_1 = []
education_set_2 = []
for idx, item in df['education'].items():
    if item <= 4:
        education_set_1.append(idx)
    else:
        education_set_2.append(idx)
print(len(education_set_2) - len(education_set_1))
income_set_1 = []
income_set_2 = []
for idx, item in df['income'].items():
    if item <= 30000:
        income_set_1.append(idx)
    else:
        income_set_2.append(idx)
print(len(income_set_2) - len(income_set_1))

dev_set_splits = {"article_id_set_1": article_id_set_1,
                  "article_id_set_2": article_id_set_2,
                  "gender_set_1": gender_set_1,
                  "gender_set_2": gender_set_2,
                  "age_set_1": age_set_1,
                  "age_set_2": age_set_2,
                  "education_set_1": education_set_1,
                  "education_set_2": education_set_2,
                  "income_set_1": income_set_1,
                  "income_set_2": income_set_2,
                  }

emp_results, dis_results, emo_results = {}, {}, {}
for split_name in dev_set_splits.keys():
    split_set = dev_set_splits[split_name]
    empathy_result = regression_ensemble_sweep(split=split_set, choices='sweep', task='empathy')
    distress_result = regression_ensemble_sweep(split=split_set, choices='sweep', task='distress')
    emotion_result = classification_ensemble_sweep(split=split_set, choices='sweep')
    emp_results[split_name] = [item['ensemble_file_names'] for item in empathy_result[:100]]
    dis_results[split_name] = [item['ensemble_file_names'] for item in distress_result[:100]]
    emo_results[split_name] = [item['ensemble_file_names'] for item in emotion_result[:100]]


with open("./result/empathy_100.json", 'w') as f:
    json.dump(emp_results, f)
with open("./result/distress_100.json", 'w') as f:
    json.dump(dis_results, f)
with open("./result/emotion_100.json", 'w') as f:
    json.dump(emo_results, f)
