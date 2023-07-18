import os
import argparse
import json
import numpy as np
from itertools import combinations
from math import sqrt
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

def pearsonr(x, y):
	"""
	Calculates a Pearson correlation coefficient. 
	"""
	assert len(x) == len(y), 'Prediction and gold standard does not have the same length...'

	xm = sum(x)/len(x)
	ym = sum(y)/len(y)

	xn = [k-xm for k in x]
	yn = [k-ym for k in y]

	r = 0 
	r_den_x = 0
	r_den_y = 0
	for xn_val, yn_val in zip(xn, yn):
		r += xn_val*yn_val
		r_den_x += xn_val*xn_val
		r_den_y += yn_val*yn_val

	r_den = sqrt(r_den_x*r_den_y)

	if r_den:
		r = r / r_den
	else:
		r = 0

	# Presumably, if abs(r) > 1, then it is only some small artifact of floating
	# point arithmetic.
	r = max(min(r, 1.0), -1.0)

	return round(r, 4)

def calculate_pearson(gold, prediction):
	"""
	gold/prediction are a list of lists [ emp pred , distress pred ]
	"""

	# converting to float
	gold = [float(k) for k in gold]
	prediction = [float(k) for k in prediction]

	return pearsonr(gold, prediction)

def calculatePRF_MLabel(y_true, y_pred, need_report=False):
    """
    gold/prediction list of list of emo predictions 
    """
    # initialize counters
    # labels = set(gold+prediction)
    to_round = 4
    # microprecision, microrecall, microf, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    macroF = f1_score(y_true, y_pred, average='macro')
    # accuracy = jaccard_score(y_true, y_pred, average='micro')
    if need_report:
        report = classification_report(y_true, y_pred)
        return round(macroF,to_round), report    
    return round(macroF,to_round)



def regression_ensemble_sweep(task='empathy', post_process=True, reduction='mean'):
    dir_path = f"./{task}/dev/"
    gold_dev_file_path = "/users10/zjli/workspace/WASSA/new_data/2023/dev.json"
    with open(gold_dev_file_path) as f:
        gold_dev_results = json.load(f)
    gold = [item[task] for item in gold_dev_results]
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
            all_pred_results.append({'pre_result':pre_result, 'file_name': file_name})
    for i in range(1, len(all_pred_results)+1): # 组合个数选取
        for combination in combinations(all_pred_results, i): # 遍历n中选i个的所有组合
            all_combined_results.append(combination) # 一种组合，即i组预测结果
    
    for combined_result in tqdm(all_combined_results): #每一种组合
        if reduction == 'mean':
            ensemble_result = [pred_result['pre_result'] for pred_result in combined_result]
            ensemble_result_array = np.mean(np.array(ensemble_result), axis=0)
            ensemble_file_names = [pred_result['file_name'] for pred_result in combined_result]
            pearson = calculate_pearson(gold, ensemble_result_array)
            final_results.append({
                "ensemble_pred_result": ensemble_result_array,
                "ensemble_file_names": ensemble_file_names,
                "ensemble_metric": pearson
            })
    final_results = sorted(final_results, key=lambda k: k['ensemble_metric'], reverse=True)
    better_combinations = []
    if task == 'empathy':
        threshold = 0.6433
    elif task == 'distress':
        threshold = 0.6346
    for item in final_results:
        if item["ensemble_metric"] > threshold:
            better_combinations.append({'ensemble_file_names': item['ensemble_file_names'], 'ensemble_metric': item['ensemble_metric']})
    print(len(better_combinations))
    print(len(final_results))
    with open(f'./{task}/better_combinations.json', 'w') as f:
        json.dump(better_combinations, f)
    
def classification_ensemble_sweep(split, task='emotion', post_process=True, reduction='label_mean'):
    dir_path = f"./{task}/{split}"
    gold_dev_file_path = "/users10/zjli/workspace/WASSA/new_data/2023/dev.json"
    with open(gold_dev_file_path) as f:
        gold_dev_results = json.load(f)
    gold = [item[task] for item in gold_dev_results]
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
    for i in range(1, len(all_pred_results)+1): # 组合个数选取
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
            metric = calculatePRF_MLabel(gold, ensemble_label_result_array)
            final_results.append({
                "ensemble_pred_result": ensemble_label_result_array,
                "ensemble_file_names": ensemble_file_names,
                "ensemble_metric": metric
            })
            pass
        else:
            raise NotImplementedError
    final_results = sorted(final_results, key=lambda k: k['ensemble_metric'], reverse=True)
    print(final_results[:3])

def ensemble_result(empathy_file_names, distress_file_names, emotion_file_names, emo_reduction='label_mean',  split='dev'):
    if not empathy_file_names or not distress_file_names or not emotion_file_names:
        raise ValueError("empathy_file_names or distress_file_names or emotion_file_names can not be empty")
    
    empathy_ensemble_result = []
    distress_ensemble_result = []
    for file_name in empathy_file_names:
        e_result = []
        with open(os.path.join(f'./empathy/{split}', file_name)) as f:
            for line in f.readlines():
                line = json.loads(line)
                if line[0] < 1.0:
                    line[0] = 1.0
                elif line[0] > 7.0:
                    line[0] = 7.0
                e_result.extend(line)
        empathy_ensemble_result.append(e_result)
    empathy_ensemble_result_array = np.mean(np.array(empathy_ensemble_result), axis=0)

    for file_name in distress_file_names:
        d_result = []
        with open(os.path.join(f'./distress/{split}', file_name)) as f:
            for line in f.readlines():
                line = json.loads(line)
                if line[0] < 1.0:
                    line[0] = 1.0
                elif line[0] > 7.0:
                    line[0] = 7.0
                d_result.extend(line)
        distress_ensemble_result.append(d_result)
    distress_ensemble_result_array = np.mean(np.array(distress_ensemble_result), axis=0)           
    
    with open(f'./result/{split}/predictions_EMP.tsv', 'w') as f:
        for emp, dis in zip(empathy_ensemble_result_array, distress_ensemble_result_array):
            f.write(f"{emp}\t{dis}")
            f.write('\n')
        
    emotion_ensemble_label_result = []
    emotion_ensemble_prob_result = []

    for file_name in emotion_file_names:
        pred_prob = []
        pred_label = []
        with open(os.path.join(f'./emotion/{split}', file_name)) as f:
            for line in f.readlines():
                line = json.loads(line)
                pred_prob.append(line['prob'])
                pred_label.append(line['p_label'])
        emotion_ensemble_prob_result.append(pred_prob)
        emotion_ensemble_label_result.append(pred_label)
        
    emotion_ensemble_prob_result_array = np.array(emotion_ensemble_prob_result)
    emotion_ensemble_prob_result_array = np.apply_along_axis(lambda x: np.mean(x), axis=0, arr=emotion_ensemble_prob_result_array)
    if emo_reduction == 'label_mean':
        emotion_ensemble_label_result_array = np.array(emotion_ensemble_label_result)
        emotion_ensemble_label_result_array = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=emotion_ensemble_label_result_array)

        for idx, label_item in enumerate(emotion_ensemble_label_result_array):
            if sum(label_item) == 0:
                max_index = np.argmax(emotion_ensemble_prob_result_array[idx])
                emotion_ensemble_label_result_array[idx][max_index] = 1
    elif emo_reduction == 'prob_mean':
        emotion_ensemble_label_result_array = np.where(emotion_ensemble_prob_result_array > 0.5, 1, 0)
        for idx, label_item in enumerate(emotion_ensemble_label_result_array):
            if sum(label_item) == 0:
                max_index = np.argmax(emotion_ensemble_prob_result_array[idx])
                emotion_ensemble_label_result_array[idx][max_index] = 1
    
    emotion_list = ['Anger', 'Disgust', 'Fear', 'Hope', 'Joy', 'Neutral', 'Sadness', 'Surprise']
    with open(f'./result/{split}/predictions_EMO.tsv', 'w') as f:
        for emo in emotion_ensemble_label_result_array:
            emotion = '/'.join([emotion_list[idx] for idx, e in enumerate(emo) if e == 1])
            f.write(f"{emotion}")
            f.write('\n')
    
# regression_ensemble_sweep(task='distress')
# classification_ensemble_sweep(reduction='prob_mean', post_process=False)

if __name__ == '__main__':
    # regression_ensemble_sweep(task='distress')
    # regression_ensemble_sweep(task='empathy')
    # classification_ensemble_sweep(split='dev', reduction='label_mean', post_process=True)
    # classification_ensemble_sweep(split='dev', reduction='prob_mean', post_process=True)

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--split', help='Description of file argument')
    args = parser.parse_args()

    split = args.split
    if split == 'dev':
        empathy_file_names = [
            "MT_3_pearson_0.6322.json", 
            "roberta-base_14_pearson_0.6367.json", 
            "roberta-base_9_pearson_0.6433.json"]
        distress_file_names = [
            "MT_1_pearson_0.626.json",
            "roberta-base_0_pearson_0.6346.json",
        ]
        emotion_file_names = [
            "0_macro_F_0.5851.json",
            "11_macro_F_0.6095.json",
            "3_macro_F_0.6087.json",
            "6_macro_F_0.588.json",
            "9_macro_F_0.6178.json",
            "13_macro_F_0.5891.json",
            "14_macro_F_0.6087.json",
        ]
    # elif split == 'test' or split == '2022_test':
    #     empathy_file_names = [
    #         "MT_3.json", 
    #         "roberta-base_14.json", 
    #         "roberta-base_9.json"]
    #     distress_file_names = [
    #         "MT_1.json",
    #         "roberta-base_0.json",
    #     ]
    #     emotion_file_names = [
    #         "0.json",
    #         "11.json",
    #         "3.json",
    #         "6.json",
    #         "9.json",
    #         "13.json",
    #         "14.json",
    #     ]
    elif split == 'test' or split == '2022_test':
        empathy_file_names = [
            "roberta-base_1.json",
            "roberta-base_13.json"]
        distress_file_names = [
            "roberta-base_1.json",
            "MT_1.json",
        ]
        emotion_file_names = [
            "0.json",
            "11.json",
            "3.json",
            "6.json",
            "9.json",

        ]
        
    ensemble_result(empathy_file_names, distress_file_names, emotion_file_names, emo_reduction='label_mean',  split=split)
