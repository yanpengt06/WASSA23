import gc
import math
import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from src.model import PLMRegressor, PLMVoteRegressor, PLMLSTMRegressor, PLMMultiTaskRegressor, PLMLSTMMultiHeadAttentionClassifier
from src.utils import set_logger, calculate_pearson, calculatePRF_MLabel, EarlyStopping, compute_kl_loss


class Trainer(object):
    def __init__(self, criterion, config, tokenizer_size,
                 train_loader, dev_loader, test_loader=None):
        # 模型
        device, device_ids = config.device, config.device_ids
        print(device)
        print(device_ids)
        if config.model_name == "PLMVoteRegressor":
            self.model = PLMVoteRegressor(config)
            self.model.plm.resize_token_embeddings(tokenizer_size)
        elif config.model_name == "PLMLSTMRegressor":
            self.model = PLMLSTMRegressor(config)
            optimizer_grouped_parameters = [
                {
                    "params": self.model.output.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.lstm.parameters(),
                    "lr": config["lstm_learning_rate"]
                },
                {
                    "params": self.model.plm.parameters(),
                    "lr": config["plm_learning_rate"]
                }
            ]
        elif config.model_name == "PLMRegressor":
            self.model = PLMRegressor(config)
            optimizer_grouped_parameters = [
                {
                    "params": self.model.output.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.plm.parameters(),
                    "lr": config["plm_learning_rate"]
                }
            ]
        elif config.model_name == "PLMMultiTaskRegressor":
            self.model = PLMMultiTaskRegressor(config)
            optimizer_grouped_parameters = [
                {
                    "params": self.model.dis_output.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.emp_output.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.plm.parameters(),
                    "lr": config["plm_learning_rate"]
                }
            ]
        elif config.model_name == "PLMLSTMMultiHeadAttentionClassifier":
            self.model = PLMLSTMMultiHeadAttentionClassifier(config)
            optimizer_grouped_parameters = [
                {
                    "params": self.model.output.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.lstm.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.attention.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.plm.parameters(),
                    "lr": config["plm_learning_rate"]
                }
            ]

        self.model = self.model.to(device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=device_ids)
            self.model = self.model.cuda(device=device_ids[0])
            optimizer_grouped_parameters = [
                {
                    "params": self.model.module.plm.parameters(),
                    "lr": config["plm_learning_rate"]
                }
            ]
            if config.model_name == "PLMLSTMRegressor":
                optimizer_grouped_parameters.extend([{
                    "params": self.model.module.lstm.parameters(),
                    "lr": config["lstm_learning_rate"]
                },
                {
                    "params": self.model.module.output.parameters(),
                    "lr": config["other_learning_rate"]
                }])
            elif config.model_name == "PLMMultiTaskRegressor":
                optimizer_grouped_parameters.extend([{
                    "params": self.model.module.emp_output.parameters(),
                    "lr": config["other_learning_rate"]
                },
                {
                    "params": self.model.module.dis_output.parameters(),
                    "lr": config["other_learning_rate"]
                }])
        if config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        elif config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)
            # for param_group in self.optimizer.param_groups:
            #     print("Learning rate for parameters:", param_group['lr'])
            #     print("betas for parameters:", param_group['betas'])
            #     print("eps for parameters:", param_group['eps'])
            #     print("weight_decay for parameters:", param_group['weight_decay'])
            #     print("amsgrad for parameters:", param_group['amsgrad'])
        # 参数初始化
        for n, p in self.model.named_parameters():
            if p.requires_grad and 'plm' not in n:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        self.criterion = criterion
        self.early_stopping = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta)

        self.device = device
        self.config = config

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.global_step = 1
        self.accumulation_steps = self.config["accumulation_steps"]
        self.epochs = self.config["epochs"]
        self.log_interval = self.config["log_interval"]
        self.total_step = len(self.train_loader) * self.epochs
        # self.lr_scheduler = None
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                            int(self.total_step *
                                                                config.warmup_proportion),
                                                            self.total_step)

        self.save_model_name = self.config["save_model_name"]
        self.save_model_dir = self.config["save_model_dir"]

        self.best_valid_metric = 0.0

        if not os.path.exists(self.config["log_dir"]):
            os.makedirs(self.config["log_dir"])
        self.logger = set_logger(os.path.join(
            self.config["log_dir"], self.config["log_save_name"]))
        self.logger.info(self.config.__dict__)

        resume = self.config["resume"]
        if resume:
            self._resume_checkpoint(resume_path=resume)

    def _train_loop(self, epoch):
        self.model.train()
        running_loss = 0.
        pred, true = None, None
        dis_pred, dis_true = None, None
        emp_pred, emp_true = None, None
        for batch, data in enumerate(self.train_loader):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)
            logits = self.model(data["input_ids"], data["attention_mask"])
            
            if self.config.task == "MT_dis_emp":
                criterion = nn.MSELoss()
                dis_logits, emp_logits = logits
                dis_labels, emp_labels = data['distress'], data['empathy']
                if dis_pred is not None:
                    dis_pred = torch.cat([dis_pred, dis_logits], dim=0)
                else:
                    dis_pred = dis_logits
                if dis_true is not None:
                    dis_true = torch.cat([dis_true, dis_labels], dim=0)
                else:
                    dis_true = dis_labels

                dis_loss = criterion(dis_logits, dis_labels)
                
                if emp_pred is not None:
                    emp_pred = torch.cat([emp_pred, emp_logits], dim=0)
                else:
                    emp_pred = emp_logits
                if emp_true is not None:
                    emp_true = torch.cat([emp_true, emp_labels], dim=0)
                else:
                    emp_true = emp_labels
                emp_loss = criterion(emp_logits, emp_labels)

                # loss = criterion(logits, labels)
                # dis_pred, dis_true, dis_loss = self._cal_loss(dis_pred, dis_true, dis_logits, dis_labels, nn.MSELoss())
                # emp_pred, emp_true, emp_loss = self._cal_loss(emp_pred, emp_true, emp_logits, emp_labels, nn.MSELoss())
                if self.config.mt_adaptive_weight:
                    loss = self.criterion(dis_loss, emp_loss)
                else:
                    loss = dis_loss + emp_loss
            else:
                labels = data[self.config.task]
                pred, true, loss = self._cal_loss(pred, true, logits, labels, self.criterion)

            if self.config['r_drop']:

                logits2 = self.model(data["input_ids"], data["attention_mask"])
                # cross entropy loss for classifier
                _, _, loss2 = self._cal_loss(None, None, logits2, labels, self.criterion)
                ce_loss = loss + loss2

                kl_loss = compute_kl_loss(logits, logits2)

                # carefully choose hyper-parameters
                loss = ce_loss + self.config['r_drop_alpha'] * kl_loss
            loss = loss / self.accumulation_steps
            loss.backward()
            if (batch+1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                # gc.collect()
                
            running_loss += loss.data.item()

            self.global_step += 1
            if self.global_step % self.log_interval == 0:
                print(
                    "[Train] epoch:{} step:{}/{} loss:{:.6f}".
                    format(epoch, self.global_step,
                           self.total_step, running_loss/(batch+1))
                )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        if self.config.task in ["distress", "empathy"]:
            pearson = calculate_pearson(pred.squeeze(
                1).cpu().tolist(), true.squeeze(1).cpu().tolist())

            self.logger.info(
                "[Train] epoch:{} step:{}/{} loss:{:.6f}, pearson:{:.6f}"
                .format(epoch, self.global_step, self.total_step, running_loss/(batch+1), pearson)
            )
        elif self.config.task == "MT_dis_emp":
            dis_pearson = calculate_pearson(dis_pred.squeeze(
                1).cpu().tolist(), dis_true.squeeze(1).cpu().tolist())
            emp_pearson = calculate_pearson(emp_pred.squeeze(
                1).cpu().tolist(), emp_true.squeeze(1).cpu().tolist())
            self.logger.info(
                "[Train] epoch:{} step:{}/{} loss:{:.6f}, dis_pearson:{:.6f}, emp_pearson:{:.6f}"
                .format(epoch, self.global_step, self.total_step, running_loss/(batch+1), dis_pearson, emp_pearson)
            )
        elif self.config.task in ["emotion"]:
            _, microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy = \
                calculatePRF_MLabel(true.cpu().detach().numpy(),
                                    pred.cpu().detach().numpy())

            self.logger.info(
                "[Train] epoch:{} step:{}/{} loss:{:.6f}, macro_f:{:.4f}, macro_p:{:.4f}, macro_r:{:.4f}, \
                    micro_f:{:.4f}, micro_p:{:.4f}, micro_r:{:.4f}, accuracy:{:.4f}"
                .format(epoch, self.global_step, self.total_step, running_loss/(batch+1), macroF, macroprecision,
                        macrorecall, microf, microprecision, microrecall, accuracy))

    def _cal_loss(self, pred, true, logits, labels, criterion):
        if pred is not None:
            pred = torch.cat([pred, logits], dim=0)
        else:
            pred = logits
        if true is not None:
            true = torch.cat([true, labels], dim=0)
        else:
            true = labels

        loss = criterion(logits, labels)
        return pred,true,loss

    def _valid_loop(self, epoch):
        self.model.eval()
        running_loss = 0.
        pred, true = None, None
        dis_pred, dis_true = None, None
        emp_pred, emp_true = None, None
        with torch.no_grad():
            for data in tqdm(self.dev_loader):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)
                
                logits = self.model(
                    data["input_ids"], data["attention_mask"])

                if self.config.task == "MT_dis_emp":
                    criterion = nn.MSELoss()
                    dis_logits, emp_logits = logits
                    dis_labels, emp_labels = data['distress'], data['empathy']
                    if dis_pred is not None:
                        dis_pred = torch.cat([dis_pred, dis_logits], dim=0)
                    else:
                        dis_pred = dis_logits
                    if dis_true is not None:
                        dis_true = torch.cat([dis_true, dis_labels], dim=0)
                    else:
                        dis_true = dis_labels

                    dis_loss = criterion(dis_logits, dis_labels)
                    
                    if emp_pred is not None:
                        emp_pred = torch.cat([emp_pred, emp_logits], dim=0)
                    else:
                        emp_pred = emp_logits
                    if emp_true is not None:
                        emp_true = torch.cat([emp_true, emp_labels], dim=0)
                    else:
                        emp_true = emp_labels
                    emp_loss = criterion(emp_logits, emp_labels)

                    # loss = criterion(logits, labels)
                    # dis_pred, dis_true, dis_loss = self._cal_loss(dis_pred, dis_true, dis_logits, dis_labels, nn.MSELoss())
                    # emp_pred, emp_true, emp_loss = self._cal_loss(emp_pred, emp_true, emp_logits, emp_labels, nn.MSELoss())
                    if self.config.mt_adaptive_weight:
                        loss = self.criterion(dis_loss, emp_loss)
                    else:
                        loss = dis_loss + emp_loss
                else:
                    labels = data[self.config.task]
                    pred, true, loss = self._cal_loss(pred, true, logits, labels, self.criterion)

                running_loss += loss.data.item()
        metric = 0
        if self.config.task in ["distress", "empathy"]:
            pearson = calculate_pearson(pred.squeeze(
                1).cpu().tolist(), true.squeeze(1).cpu().tolist())
            metric = pearson
            self.logger.info(
                "[Valid] epoch:{}, loss:{:.6f}, pearson:{:.6f}"
                .format(epoch, running_loss/len(self.dev_loader), pearson)
            )
        elif self.config.task == "MT_dis_emp":
            dis_pearson = calculate_pearson(dis_pred.squeeze(
                1).cpu().tolist(), dis_true.squeeze(1).cpu().tolist())
            emp_pearson = calculate_pearson(emp_pred.squeeze(
                1).cpu().tolist(), emp_true.squeeze(1).cpu().tolist())
            metric = (dis_pearson + emp_pearson) / 2
            self.logger.info(
                "[Valid] epoch:{} loss:{:.6f}, dis_pearson:{:.6f}, emp_pearson:{:.6f}"
                .format(epoch, running_loss/len(self.dev_loader), dis_pearson, emp_pearson)
            )
        elif self.config.task in ["emotion"]:
            report, microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy \
                = calculatePRF_MLabel(true.cpu().detach().numpy(), pred.cpu().detach().numpy())
            metric = macroF
            self.logger.info(
                "[Valid] epoch:{}, loss:{:.6f}, macro_f:{:.4f}, macro_p:{:.4f}, macro_r:{:.4f}, \
micro_f:{:.4f}, micro_p:{:.4f}, micro_r:{:.4f}, accuracy:{:.4f}, \n report:{}"
                .format(epoch, running_loss/len(self.dev_loader), macroF, macroprecision,
                        macrorecall, microf, microprecision, microrecall, accuracy, report)
            )

        return running_loss/len(self.dev_loader), metric

    def _save_checkpoint(self, state):
        """
        Saving checkpoints
        """
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        checkpoint_path = os.path.join(
            self.save_model_dir, self.save_model_name)
        torch.save(state, checkpoint_path)
        self.logger.info(
            f"Saving current best model: {self.save_model_name} ...")

    def _resume_checkpoint(self, resume_path):
        """
        resume from saved checkpoints
        """
        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']))

    def train(self):
        for epoch in range(self.epochs):
            self._train_loop(epoch)
            dev_loss, metric = self._valid_loop(epoch)
            # self.early_stopping(dev_loss)
            self.early_stopping(metric, reverse=True)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
            if metric > self.best_valid_metric:
                self.best_valid_metric = metric
                if self.config.save_model:
                    checkpoint = {
                        "epoch": epoch,
                        "loss": dev_loss,
                        "metric": metric,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()
                    }
                    self._save_checkpoint(checkpoint)
            metrics = {"epoch": epoch,
                       "valid_loss": dev_loss,
                       "valid_metric": metric,
                       "best_valid_metric": self.best_valid_metric}
            if self.config.use_wandb:
                wandb.log(metrics)

        # checkpoint = {
        #             "epoch": epoch,
        #             "loss": dev_loss,
        #             "pearson": pearson,
        #             "state_dict": self.model.state_dict(),
        #             "optimizer": self.optimizer.state_dict()
        #         }
        # if not os.path.exists(self.save_model_dir):
        #     os.makedirs(self.save_model_dir)
        # checkpoint_path = os.path.join(self.save_model_dir, "last_but_not_"+self.save_model_name)
        # torch.save(checkpoint, checkpoint_path)
        # self.logger.info(f"Saving last epoch model: last_but_not_{self.save_model_name} ...")

    def evaluate(self):
        checkpoint_path = os.path.join(
            self.save_model_dir, self.save_model_name)
        self._resume_checkpoint(checkpoint_path)
        self.model.eval()
        running_loss = 0.
        pred, true = None, None
        dis_pred, dis_true = None, None
        emp_pred, emp_true = None, None
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)

                logits = self.model(
                        data["input_ids"], data["attention_mask"])

                if self.config.task == "MT_dis_emp":
                    dis_logits, emp_logits = logits
                    dis_labels, emp_labels = data['distress'], data['empathy']
                    dis_pred, dis_true, dis_loss = self._cal_loss(dis_pred, dis_true, dis_logits, dis_labels, nn.MSELoss())
                    emp_pred, emp_true, emp_loss = self._cal_loss(emp_pred, emp_true, emp_logits, emp_labels, nn.MSELoss())
                    if self.config.mt_adaptive_weight:
                        loss = self.criterion(dis_loss, emp_loss)
                    else:
                        loss = dis_loss + emp_loss
                else:
                    labels = data[self.config.task]
                    pred, true, loss = self._cal_loss(pred, true, logits, labels, self.criterion)

                loss = self.criterion(logits, labels)

                running_loss += loss.data.item()

        metric = 0
        if self.config.task in ["distress", "empathy"]:
            pearson = calculate_pearson(pred.squeeze(
                1).cpu().tolist(), true.squeeze(1).cpu().tolist())
            metric = pearson
            self.logger.info(
                "[Test] loss:{:.6f}, pearson:{:.6f}"
                .format(running_loss/len(self.test_loader), pearson)
            )
        elif self.config.task == "MT_dis_emp":
            dis_pearson = calculate_pearson(dis_pred.squeeze(
                1).cpu().tolist(), dis_true.squeeze(1).cpu().tolist())
            emp_pearson = calculate_pearson(emp_pred.squeeze(
                1).cpu().tolist(), emp_true.squeeze(1).cpu().tolist())
            metric = (dis_pearson + emp_pearson) / 2
            self.logger.info(
                "[Test] loss:{:.6f}, dis_pearson:{:.6f}, emp_pearson:{:.6f}"
                .format(running_loss/len(self.test_loader), dis_pearson, emp_pearson)
            )
        elif self.config.task in ["emotion"]:
            _, microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy \
                = calculatePRF_MLabel(true.cpu().detach().numpy(), pred.cpu().detach().numpy())
            metric = macroF
            self.logger.info(
                "[Test] loss:{:.6f}, macro_f:{:.4f}, macro_p:{:.4f}, macro_r:{:.4f}, \
                    micro_f:{:.4f}, micro_p:{:.4f}, micro_r:{:.4f}, accuracy:{:.4f}"
                .format(running_loss/len(self.test_loader), macroF, macroprecision,
                        macrorecall, microf, microprecision, microrecall, accuracy)
            )

        return metric
