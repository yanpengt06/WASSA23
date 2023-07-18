import copy
import os
import yaml
import re
from pprint import pprint
# def log_analysis(log_path):
#     valid_pearsons = []
#     test_pearson = 0
#     with open(log_path) as f:
#         for line in f.readlines():
#             if "Valid" in line:
#                 line = line.split("pearson:")
#                 valid_pearsons.append(float(line[-1]))
#             if "Test" in line:
#                 line = line.split("pearson:")
#                 test_pearson = float(line[-1])
#             if "{'_items': " in line:
#                 line = line.split("{'_items': ")
#                 para_str = line[1].split(", 'task':")[0]+'}'
#                 para = eval(para_str)

#     return max(valid_pearsons) if valid_pearsons else 0, test_pearson, para

# def main(seed, task, model_name):
#     res_dict = []
#     chosed_paras = []
#     tmp_dict = {"program": "/users10/zjli/workspace/WASSA/task2/run.py", "method": "grid", "project": "regression", "name":f"{task}_{model_name}", \
#         "metric":{"goal": "maximize", "name": "best_valid_pearson"}, "parameters":{\
#             "seed":{"values":[5678, 9101112, 13141516, 17181920]},
#             "task":{"values":["empathy"]},
#             "model_name":{"values":[1]},
#             "batch_size":{"values":[]},
#             "plm_learning_rate":{"values":[]},
#             "other_learning_rate":{"values":[]},
#             "dropout":{"values":[]}}}
#     # paras = []
#     filelist = os.listdir(f"./logs/{seed}/{task}/{model_name}/")
#     for item in filelist:
#         if item.endswith(".log"):
#             vp, tp, para = log_analysis(f"./logs/{seed}/{task}/{model_name}/{item}")
#             res_dict.append({"log_name": item, "valid_p":vp, "test_p":tp, "batch_size":para["batch_size"], \
#                 "dropout":para["dropout"], "plm_learning_rate":para["plm_learning_rate"], \
#                     "other_learning_rate": para["other_learning_rate"]})
#     res_dict.sort(key = lambda p: p["valid_p"], reverse=True)
#     sorted_by_v = res_dict[:10]
#     print("valid最高")
#     print(res_dict[0])
#     chosed_paras.append(res_dict[0])
#     res_dict.sort(key = lambda p: p["test_p"], reverse=True)
#     sorted_by_t = res_dict[:10]
#     print("test最高")
#     print(res_dict[0])
#     chosed_paras.append(res_dict[0])
#     print("valid和test前10的交集")
#     print([v for v in sorted_by_t if v in sorted_by_v])
#     chosed_paras.extend([v for v in sorted_by_t if v in sorted_by_v])
#     # print(chosed_paras)
#     for para in chosed_paras:
#         tmp_dict["parameters"]["batch_size"] = [para["batch_size"]]
#         tmp_dict["parameters"]["plm_learning_rate"] = [para["plm_learning_rate"]]
#         tmp_dict["parameters"]["other_learning_rate"] = [para["other_learning_rate"]]
#         tmp_dict["parameters"]["dropout"] = [para["dropout"]]
#         d_dict = {0:'0', 0.3:"03", 0.5:"05"}
#         # paras.append(copy.deepcopy(tmp_dict))
#         # a = str(para["other_learning_rate"])[-1]
#         a = para["other_learning_rate"]
#         # b = str(para["plm_learning_rate"])[0]
#         b = para["plm_learning_rate"]
#         # c = d_dict[para["dropout"]]
#         c = para["dropout"]
#         d = para["batch_size"]
#         mn = 0 if model_name=="roberta-base" else 1
#         for _seed in [5678, 9101112, 13141516, 17181920]:
#             print(f"python run.py --task {task} --model_name {mn} --batch_size {d} --dropout {c} --plm_learning_rate {b} --other_learning_rate {a} --seed {_seed}")
#         print("\n")
#         # with open(f"./config/{task}/{model_name}/bsz_{d}_d_{c}_plr_{b}_olr_{a}.yaml", 'w') as f:
#         #     yaml.dump(copy.deepcopy(tmp_dict), f)
#         # break
#     # for para in paras:
#     #     print(para)


def re_log(log_path):
    valid_loss_pattern = re.compile(
        r'\[Valid\] epoch:(\d+), loss:(\d+\.\d+), macro_f:(\d+\.\d+), macro_p:(\d+\.\d+), '+
        'macro_r:(\d+\.\d+), micro_f:(\d+\.\d+), micro_p:(\d+\.\d+), micro_r:(\d+\.\d+), accuracy:(\d+\.\d+)')

    valid_loss = []

    lines = []
    with open(log_path, 'r') as f:
        for line in f:
            valid_match = valid_loss_pattern.search(line)
            if valid_match:
                
                epoch = int(valid_match.group(1))
                loss = float(valid_match.group(2))
                macro_f = float(valid_match.group(3))
                macro_p = float(valid_match.group(4))
                macro_r = float(valid_match.group(5))
                micro_f = float(valid_match.group(6))
                micro_p = float(valid_match.group(7))
                micro_r = float(valid_match.group(8))
                accuracy = float(valid_match.group(9))
                valid_loss.append(
                    (epoch, loss, macro_f, macro_p, macro_r, micro_f, micro_p, micro_r, accuracy))
                if macro_f > 0.58 and line not in lines:
                    lines.append(line)
        
    return lines

def re_regression_log(log_path, threshold=0.58):
    valid_loss_pattern = re.compile(
        r'\[Valid\] epoch:(\d+), loss:(\d+\.\d+), pearson:(\d+\.\d+)')

    valid_loss = []

    lines = []
    with open(log_path, 'r') as f:
        for line in f:
            valid_match = valid_loss_pattern.search(line)
            if valid_match:
                epoch = int(valid_match.group(1))
                loss = float(valid_match.group(2))
                pearson = float(valid_match.group(3))

                valid_loss.append(
                    (epoch, loss, pearson))
                if pearson > threshold and line not in lines:
                    lines.append(line)
        
    return lines

def re_mt_regression_log(log_path, threshold1=0.58, threshold2=0.62):
    # valid_loss_pattern = re.compile(
    #     r'\[Valid\] epoch:(\d+), loss:(\d+\.\d+), pearson:(\d+\.\d+)')
    valid_loss_pattern = re.compile(
        r'\[Valid\] epoch:(\d+) loss:(\d+\.\d+), dis_pearson:(\d+\.\d+), emp_pearson:(\d+\.\d+)')

    valid_loss = []

    lines = []
    with open(log_path, 'r') as f:
        for line in f:
            valid_match = valid_loss_pattern.search(line)
            if valid_match:
                epoch = int(valid_match.group(1))
                loss = float(valid_match.group(2))
                dis_pearson = float(valid_match.group(3))
                emp_pearson = float(valid_match.group(4))
                valid_loss.append(
                    (epoch, loss, dis_pearson, emp_pearson))
                if (dis_pearson > threshold1 or emp_pearson > threshold2) and line not in lines:
                    print(dis_pearson, emp_pearson)
                    print(log_path)
                    lines.append(line)
        
    return lines

def parse_param_from_log(log_path, metric_log_line):
    with open(log_path) as f:
        lines = f.readlines()
    for i in range(len(lines)-1, -1, -1):
        if lines[i] == metric_log_line:
            break
    for j in range(i, -1, -1):
        if '__init__' in lines[j]:
            try:
                parameters = eval(''.join(('{'+lines[j].split('{')[1]).split("'device': device(type='cuda', index=0),")))
            except:
                parameters = eval(lines[j].split('trainer.py - __init__')[1][9:].split(", '_locked': ")[0]+'}')['_items']
            break
    return parameters

if __name__ == "__main__":    

    dir_name = '/users10/zjli/workspace/WASSA/task2&3/logs/1234/MT_dis_emp/roberta-base/PLMMultiTaskRegressor'

    dirs = os.listdir(dir_name)
    log_params = []
    for log_name in dirs:
        lines = re_mt_regression_log(f'{dir_name}/{log_name}', 0.60, 0.63)
        params = []
        if lines != []:
            for line in lines:
                param = parse_param_from_log(f'{dir_name}/{log_name}', line)
                if param not in params:
                    params.append(param)
        if params:
            log_params.append({'log_path': f'{dir_name}/{log_name}', 'params': params})
    # pprint(log_params)
    # print(len(log_params))
    # print(sum([len(item['params']) for item in log_params]))
    
    finetune_params = [
    "dropout",
    "model_name",
    "pooling",
    "plm_learning_rate",
    "plm_model_num",
    "optimizer",
    "batch_size",
    "task",
    "num_labels",
    ]
    finetune_params.sort()
    command_lines = []
    for param in log_params:
        log_path = param['log_path']
        ps = param['params']
        for p in ps:
            com_l = []
            for fp in finetune_params:
                if fp in p.keys():
                    com_l.append(f"--{fp}={p[fp]}")
                elif fp == 'optimizer':
                    com_l.append(f"--{fp}=Adam")
                else:
                    print(fp)
            
            if f"python run.py {' '.join(com_l)}" not in command_lines:
                command_lines.append(f"python run.py {' '.join(com_l)}")
    print(len(command_lines))
    command_lines.sort()
    for item in command_lines:
        print(item+" --save_model=True")
        print('\n')
    
# if __name__ == "__main__":    
#     log_pattern = re.compile(
#         r'log_bsz_(\d+)_dr_(\d+(.\d+))_plr_(\d+)e-05_trsp_chatgpt_aug_datav(\d+)_pool_mean.log')
#     dir_name = '/users10/zjli/workspace/WASSA/task2&3/logs/1234/distress/roberta-base/PLMRegressor'

#     dirs = os.listdir(dir_name)
#     log_params = []
#     for log_name in dirs:
#         lines = re_regression_log(f'{dir_name}/{log_name}', 0.58)
#         params = []
#         if lines != []:
#             for line in lines:
#                 param = parse_param_from_log(f'{dir_name}/{log_name}', line)
#                 if param not in params:
#                     params.append(param)
#         if params:
#             log_params.append({'log_path': f'{dir_name}/{log_name}', 'params': params})
#     pprint(log_params)
#     print(len(log_params))
#     print(sum([len(item['params']) for item in log_params]))
    # print(set([key for item in log_params for param in item['params'] for key in param]))
    
    # finetune_params = [
    # "dropout",
    # "dev_split",
    # "r_drop",
    # "model_name",
    # "pooling",
    # "plm_learning_rate",
    # "plm_model_num",
    # "data_path",
    # "r_drop_alpha",
    # "optimizer",
    # "batch_size",
    # "lstm_lr_scale",
    # "loss_type",
    # "train_split",
    # "epochs",
    # "task",
    # "num_labels"
    # ]
    # finetune_params.sort()
    # command_lines = []
    # for param in log_params:
    #     log_path = param['log_path']
    #     ps = param['params']
    #     for p in ps:
    #         com_l = []
    #         for fp in finetune_params:
    #             if fp in p.keys():
    #                 com_l.append(f"--{fp}={p[fp]}")
    #             elif fp == 'r_drop':
    #                 com_l.append(f"--{fp}=False")
    #             elif fp == 'r_drop_alpha':
    #                 com_l.append(f"--{fp}=1.0")
    #             elif fp == 'lstm_lr_scale':
    #                 com_l.append(f"--{fp}=0")
    #             else:
    #                 print(fp)
            
    #         if f"python run.py {' '.join(com_l)}" not in command_lines:
    #             command_lines.append(f"python run.py {' '.join(com_l)}")
    # print(len(command_lines))
    # command_lines.sort()
    # for item in command_lines:
    #     print(item+" --save_model=True")
    #     print('\n')