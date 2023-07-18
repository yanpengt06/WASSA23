import os
import wandb
import torch.nn as nn
from src.config import ArgConfig
from src.data_loader import get_loader
from src.utils import set_seed, parse_args
from src.loss import Multilabel_Categorical_Crossentropy, AutomaticWeightedLoss
from src.trainer import Trainer


set_seed(seed=1234) # 5678 9101112 13141516 17181920

def run(config):
    print("读取训练集数据……")
    train_loader, tokenizer_size = get_loader(config, split=config.train_split, shuffle=True, demo=config.demo)
    print("读取验证集数据……")
    dev_loader, _ = get_loader(config, split=config.dev_split, shuffle=False, demo=config.demo)
    test_loader = None
    if config["do_test"]:
        print("读取测试集数据……")
        test_loader, _ = get_loader(config, split='2022_test', shuffle=False, demo=config.demo)
    # device = torch.device('cuda:0' if config['n_gpu'] > 0 else 'cpu')
    # emotions = [('Anger', 124), ('Disgust', 100), ('Fear', 33), ('Hope', 32), ('Joy', 10), ('Neutral', 240), 
    #             ('Sadness', 383), ('Surprise', 19)]
    # emotion_weight = torch.Tensor([100/e[1] for e in emotions]).to(device)
    
    # 损失函数
    if config.task in ["distress", "empathy"]:
        criterion = nn.MSELoss()
    elif config.task in ["MT_dis_emp"]:
        if config.mt_adaptive_weight:
            criterion = AutomaticWeightedLoss(num=2)
        else:
            criterion = nn.MSELoss()
    elif config.task in ["emotion"]:
        if config.loss_type == "BCE":
            criterion = nn.BCEWithLogitsLoss()
        elif config.loss_type == "MCCE":
            criterion = Multilabel_Categorical_Crossentropy()

    # trainer
    trainer = Trainer(criterion, config, tokenizer_size, train_loader, dev_loader, test_loader)
    
    # 训练
    if config["do_train"]:
        print("Train begin")
        trainer.train()
        
    # 测试
    if config["do_test"]:
        print("Test Begin")
        if not config["do_train"]:
            assert config.resume is not None, 'make sure resume is not None'
        pearson = trainer.evaluate()
        print(f"pearson: {pearson}")

if __name__ == "__main__":
    args = parse_args()
    conf = ArgConfig(**args)
    conf.do_train = True
    conf.do_test = False
    conf.update_cache = False
    conf.demo = False
    if conf.use_wandb:
        wandb.init(config=conf, mode="offline")
        conf = wandb.config
    from pprint import pprint
    pprint(conf.__dict__)
    run(conf)
