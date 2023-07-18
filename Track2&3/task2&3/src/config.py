from torch.cuda import device_count
from src.utils import prepare_device
import time
PLM_MODEL_NAMES = [
    "roberta-base", 
    "roberta-large", 
    "allenai/dsp_roberta_base_dapt_reviews_tapt_imdb_70000", 
    "/users10/zjli/workspace/WASSA/pretraining/roberta-large/100K/roberta-large_e_100_lr_1e-4/checkpoint-2600",
    "/users10/zjli/workspace/WASSA/pretraining/roberta-large/488K/roberta-large_e_100_lr_1e-4/checkpoint-12616"]

class ArgConfig(object):
    def __init__(self, **kwargs):
        
        # 模型组件
        self.task = kwargs.get('task', None)
        self.num_labels = kwargs.get('num_labels', None)
        self.vote = kwargs.get('vote', False)=='True'
        self.voter_num = kwargs.get('voter_num', 6)
        self.use_BiLSTM = kwargs.get('use_BiLSTM', False)=='True'
        self.add_attention = kwargs.get('add_attention', False)=='True'
        self.num_heads = kwargs.get('num_heads', 4)
        self.model_name = kwargs.get('model_name', 'PLMRegressor')
        self.train_split = kwargs.get('train_split', 'train')
        self.dev_split = kwargs.get('dev_split', 'dev')
        self.loss_type = kwargs.get('loss_type', "BCE")
        self.pooling = kwargs.get('pooling', None)
        self.optimizer = kwargs.get('optimizer', "AdamW")
        self.r_drop = kwargs.get('r_drop', False)=='True'
       
        # 模型超参数
        self.dropout = kwargs.get('dropout', 0)
        self.n_gpu = device_count()
        self.batch_size = kwargs.get('batch_size', 8)
        self.epochs = kwargs.get('epochs', 15)
        self.log_interval = kwargs.get('log_interval', 20)
        self.seed = kwargs.get('seed', 1234)
        self.accumulation_steps = kwargs.get('accumulation_steps', 1)
        self.warmup_proportion = kwargs.get('warmup_proportion', 0.1)
        self.patience = kwargs.get('patience', 10)
        self.min_delta = kwargs.get('min_delta', 0.001)
        self.device, self.device_ids = prepare_device(self.n_gpu)
        self.lstm_hidden_size = kwargs.get('lstm_hidden_size', 256)
        self.save_model = kwargs.get('save_model', False)=='True'
        self.mt_adaptive_weight = kwargs.get('mt_adaptive_weight', False)=='True'
        self.r_drop_alpha = kwargs.get('r_drop_alpha', 1.0)
        
        # 优化器超参数
        self.plm_learning_rate = kwargs.get('plm_learning_rate', 1e-5)
        self.other_learning_rate = self.plm_learning_rate*10
        self.lstm_lr_scale = kwargs.get('lstm_lr_scale', 0)
        self.lstm_learning_rate = self.plm_learning_rate * self.lstm_lr_scale
        
        # 路径和名称
        self.resume = None
        self.plm_model_num = kwargs.get('plm_model_num', None)
        self.plm_model_name = PLM_MODEL_NAMES[self.plm_model_num]
        self.save_model_dir = f'./checkpoints/{self.seed}/{self.task}/{self.plm_model_name.replace("/", "-")}/' + \
                                  f'{self.model_name}/'
        self.save_model_name = f'bsz_{self.batch_size}_dr_{self.dropout}_plr_{self.plm_learning_rate}' + \
                                   f'_trsp_{self.train_split}_pool_{self.pooling}_best_model_r_drop_{self.r_drop}_{time.time()}.pt'
        self.log_dir = f'./logs/{self.seed}/{self.task}/{self.plm_model_name.replace("/", "-")}/{self.model_name}/'
        self.log_save_name = f'log_bsz_{self.batch_size}_dr_{self.dropout}_plr_{self.plm_learning_rate}' + \
                                 f'_trsp_{self.train_split}_pool_{self.pooling}_r_drop_{self.r_drop}_{time.time()}.log'
        self.data_path = kwargs.get('data_path', '../new_data/2023/')
        self.cache_path = f'./cache/'
        self.result_dir = f'./result/'
        if self.plm_model_num in [0, 1]:
            self.plm_model_name = f"/users10/zjli/plm-models/english/roberta/{self.plm_model_name}"

        # 训练/测试
        self.do_train = kwargs.get('do_train', True)=='True'
        self.do_test = kwargs.get('do_test', False)=='True'
        self.demo = kwargs.get('demo', False)=='True'
        self.update_cache = kwargs.get('update_cache', False)=='True'
        self.use_wandb = kwargs.get('use_wandb', False) =='True'

    def __getitem__(self, item):
        return getattr(self, item)
