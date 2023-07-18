# coding=utf-8
import torch
import os, sys
import random
import logging
import numpy as np
from torch import nn
from torch.nn import functional as F
from math import sqrt
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, classification_report
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

def calculatePRF_MLabel(y_true, y_pred):
    """
    gold/prediction list of list of emo predictions 
    """
    # initialize counters
    # labels = set(gold+prediction)
    y_pred = 1 / (1 + np.exp(-y_pred))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    to_round = 4
    microprecision, microrecall, microf, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    macroprecision, macrorecall, macroF, s = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = jaccard_score(y_true, y_pred, average='micro')
    report = classification_report(y_true, y_pred)
    return report, round(microrecall,to_round),round(microprecision,to_round),round(microf,to_round),\
     round(macrorecall,to_round),round(macroprecision,to_round),round(macroF,to_round),round(accuracy,to_round)

# 解析命令行参数
def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        key = key.split('--')[1]
        try:
            value = int(value)
        # 如果转换失败，则尝试将 value 转换为浮点数类型
        except ValueError:
            try:
                value = float(value)
            # 如果转换失败，则保持原始字符串类型
            except ValueError:
                pass
        args[key] = value
    return args

def set_seed(seed=42):
    """
    设置随机种子，保证实验可重现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    
    
def prepare_device(n_gpu_use):
    """
    setup GPU device if available, get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There's no GPU available on this machine, "
              "training will be performed on CPU.")
        n_gpu = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU configured to use is {}."
              "but only {} are available on this machine".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def set_logger(log_path):
    """
    配置日志
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler
    # 会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(stream_handler)
    return logger

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss, reverse=False):
        if reverse:
            val_loss = -val_loss
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                

def kld_binary(y_true, y_pred):
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    epsilon = 1e-7
    y_true_pos = torch.clamp(y_true, epsilon, 1-epsilon)
    y_pred_pos = torch.clamp(y_pred, epsilon, 1-epsilon)
    y_true_neg = torch.clamp(y_true_neg, epsilon, 1-epsilon)
    y_pred_neg = torch.clamp(y_pred_neg, epsilon, 1-epsilon)
    return torch.mean(y_true_pos * torch.log(y_true_pos / y_pred_pos) + y_true_neg * torch.log(y_true_neg / y_pred_neg), axis=-1)

def compute_kl_loss(logits1, logits2):
    kl_loss1 = torch.mean(kld_binary(torch.sigmoid(logits1), torch.sigmoid(logits2)))
    kl_loss2 = torch.mean(kld_binary(torch.sigmoid(logits2), torch.sigmoid(logits1)))
    loss = (kl_loss1 + kl_loss2) / 2
    return loss

