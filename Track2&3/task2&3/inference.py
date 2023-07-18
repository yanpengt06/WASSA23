import argparse, sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from src.config import ArgConfig
from src.data_loader import get_loader
from src.model import PLMRegressor, PLMNeutralClassifier, PLMLSTMRegressor
from src.utils import calculate_pearson, prepare_device, calculatePRF_MLabel
import numpy as np
# def predict(model, tokenizer, text, device):
#     model.eval()
#     with torch.no_grad():
#         tokenized_sentence = tokenizer(text)
#         input_ids = torch.LongTensor([tokenized_sentence["input_ids"]]).to(device)
#         attention_mask = torch.LongTensor([tokenized_sentence["attention_mask"]]).to(device)
#         pred_label = model(input_ids, attention_mask)
#     return pred_label

def cls_evaluate(model, test_loader, config):
    model.eval()
    results = []
    pred, true = None, None
    with torch.no_grad():
        for data in tqdm(test_loader):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)
            
            logits = model(data["input_ids"], data["attention_mask"])
      
            labels = data[config['task']]
            
            if pred != None:
                pred = torch.cat([pred, logits], dim=0)
            else:
                pred = logits
            if true != None:
                true = torch.cat([true, labels], dim=0)
            else:
                true = labels
    
    y_pred = pred.squeeze(1).cpu()
    y_pred = 1 / (1 + np.exp(-y_pred))
    probabilty = y_pred.tolist()
    y_pred = np.where(y_pred > 0.5, 1, 0).tolist()
    
    
    
    with open(f"./pred/predictions_{conf.task}.jsonl", 'w', encoding="utf-8") as f:
        for p, l in zip(probabilty, y_pred):
            line = json.dumps({'prob': p, 'label': l})
            f.write(line)
            f.write('\n')

    pearson = calculatePRF_MLabel(true.cpu(), pred.cpu())

    return pearson

def reg_evaluate(model, test_loader, config):
    model.eval()
    results = []
    pred, true = None, None
    with torch.no_grad():
        for data in tqdm(test_loader):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)
            
            logits = model(data["input_ids"], data["attention_mask"])
            labels = data["labels"]
            
            if pred != None:
                pred = torch.cat([pred, logits], dim=0)
            else:
                pred = logits
            if true != None:
                true = torch.cat([true, labels], dim=0)
            else:
                true = labels
                
    with open(f"./pred/predictions_{conf.task}.json", 'w', encoding="utf-8") as f:
        json.dump(pred.squeeze(1).cpu().tolist(), f)
        
    pearson = calculate_pearson(pred.squeeze(1).cpu().tolist(), true.squeeze(1).cpu().tolist())
    print(pearson)
    # return pearson

if __name__ == "__main__":
    from src.utils import parse_args
    args = parse_args()
    conf = ArgConfig(**args)
    conf.update_cache = True
    # model_path = "./checkpoints/data_v1/1234/A0_A1/hfl-chinese-roberta-wwm-ext/best_1669627539.19448.pt"
    plm_model_name = conf.plm_model_name.replace("/", "-")
    # model_path = f"./checkpoints/1234/emotion/roberta-large/PLMRegressor/best_1680961375.0426128.pt"
    model_path = f"/users10/zjli/workspace/WASSA/task2&3/checkpoints/1234/emotion/roberta-large/PLMLSTMRegressor/bsz_8_dr_0.3_plr_1e-05_trsp_chatgpt_aug_datav2_pool_mean_best_model.pt"

    model = PLMLSTMRegressor(conf)
    device, device_ids = prepare_device(conf['n_gpu'])
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
    
    print("读取测试集数据……")
    test_loader, _  = get_loader(conf, split='dev', shuffle=False, demo=False)
    cls_evaluate(model, test_loader, conf)
    # print(f"pearson: {pearson}")
