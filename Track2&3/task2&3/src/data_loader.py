import json, os, pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

class MyDataset(Dataset):
    def __init__(self, config, data_path, split, demo):
        super().__init__()
        update_cache = config.update_cache
        # task = config.task
        self.config = config
        self.input_ids = []
        self.attention_mask = []
        self.dis_labels, self.emp_labels, self.emo_labels = [], [], []
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_model_name)
        if config.vote:
            self.additional_special_tokens = [f"<s{i}>" for i in range(1, config.voter_num+1)]
            print(self.additional_special_tokens)
            special_tokens_dict = {"additional_special_tokens": self.additional_special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer_size = len(self.tokenizer)
        # emotion_labels = [" anger", " disgust", " fear", " hope", " joy", " neutral", " sadness", " surprise"]
        # self.emo_token_ids = []
        # for emo in emotion_labels:
        #     self.emo_token_ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(emo)))
            
        # self.bos_token_id = self.tokenizer.bos_token_id
        # self.eos_token_id = self.tokenizer.eos_token_id
        
        self._load_data(data_path, split, update_cache, demo)
    
    def _load_data(self, data_path, split, update_cache=True, demo=False):
        if self.config.vote:
            cache_name = os.path.join(self.config["cache_path"], f"voter_{self.config.voter_num}_{split}.pkl")
        else:
            cache_name = os.path.join(self.config["cache_path"], f"{split}.pkl")
            
        if not os.path.exists(self.config["cache_path"]):
            os.makedirs(self.config["cache_path"])
            
        if os.path.exists(cache_name) and not update_cache:
            print("Loading data from cache")
            (self.input_ids,
             self.attention_mask,
             self.dis_labels,
             self.emp_labels,
             self.emo_labels
            ) = pickle.load(open(cache_name, "rb"), encoding="latin1")
        else:
            with open(data_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                
            for idx, item in enumerate(tqdm(data)):
                if idx == 30 and demo == True:
                    break
                essay = item["essay"]
                if self.config.vote:
                    essay = ''.join(self.additional_special_tokens)+essay

                # if 'test' in split:
                #     dis_label = None
                #     emp_label = None
                #     emo_label = None
                # else:
                # dis_label = item["distress"]
                # emp_label = item["empathy"]
                # emo_label = item["emotion"]
                dis_label = item.get("distress", None)
                emp_label = item.get("empathy", None)
                emo_label = item.get("emotion", None)
                
                tokenized_essay = self.tokenizer(essay)
                input_ids = tokenized_essay["input_ids"]
                attention_mask = tokenized_essay["attention_mask"]
                
                if len(input_ids) <= self.tokenizer.model_max_length:
                    self.input_ids.append(input_ids)
                    self.attention_mask.append(attention_mask)
                else:
                    self.input_ids.append(input_ids[:self.tokenizer.model_max_length-1] + [self.tokenizer.sep_token_id])
                    self.attention_mask.append(attention_mask[:self.tokenizer.model_max_length])
                self.dis_labels.append(dis_label)
                self.emp_labels.append(emp_label)
                self.emo_labels.append(emo_label)

            pickle.dump((self.input_ids, self.attention_mask, self.dis_labels, self.emp_labels, self.emo_labels), \
                open(cache_name, "wb"))

            
        print("Done")
    
    def __len__(self):
        return len(self.emo_labels)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def __getitem__(self, index):
        return (torch.LongTensor(self.input_ids[index]),
                torch.LongTensor(self.attention_mask[index]),
                torch.FloatTensor([self.dis_labels[index]]) if self.dis_labels[index] is not None else None,
                torch.FloatTensor([self.emp_labels[index]]) if self.emp_labels[index] is not None else None,
                torch.FloatTensor([self.emo_labels[index]]) if self.emo_labels[index] is not None else None,
                )
                
def collate_fn(batch):
    input_ids, attention_mask, dis_labels, emp_labels, emo_labels = zip(*batch)
        
    batch_input_ids = pad_sequence(input_ids, batch_first=True)
    batch_attention_mask = pad_sequence(attention_mask, batch_first=True)
    # batch_dis_labels = torch.cat(dis_labels)
    # batch_dis_labels = batch_dis_labels.unsqueeze(1)
    # batch_emp_labels = torch.cat(emp_labels)
    # batch_emp_labels = batch_emp_labels.unsqueeze(1)
    # batch_emo_labels = torch.cat(emo_labels)
    # batch_emo_labels = batch_emo_labels
    batch_dis_labels = torch.cat([label.unsqueeze(1) for label in dis_labels if label is not None]) \
        if any(dis_labels) else None
    batch_emp_labels = torch.cat([label.unsqueeze(1) for label in emp_labels if label is not None]) \
        if any(emp_labels) else None
    non_empty_labels = [label for label in emo_labels if label is not None]
    batch_emo_labels = torch.cat(non_empty_labels) if non_empty_labels else None
    
    batch_data = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "distress": batch_dis_labels,
        "empathy": batch_emp_labels,
        "emotion": batch_emo_labels
    }
    return batch_data


def get_loader(config, split, shuffle, demo=False):
    filename = os.path.join(config["data_path"], f"{split}.json")
    dataset = MyDataset(config, filename, split, demo)
    print(f"{split}: {dataset[0]}")
    data_loader = DataLoader(dataset,
                             batch_size=config.batch_size,
                             shuffle=shuffle,
                             collate_fn=collate_fn)
    return data_loader, dataset.tokenizer_size                   
                    