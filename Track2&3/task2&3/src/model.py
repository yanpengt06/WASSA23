import torch
import torch.nn as nn
import torch.nn.functional as F
from deprecated import deprecated
from transformers import AutoModel



class PLMRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(self.plm.config.hidden_size, config.num_labels)
        self.config = config
    
    def forward(self, input_ids, attention_mask):
        plm_outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = plm_outputs.last_hidden_state
        if self.config.pooling == 'mean':
            last_hidden_state_sum = torch.sum(last_hidden_state*attention_mask.unsqueeze(-1), 1)
            pool_output = last_hidden_state_sum / attention_mask.sum(-1).unsqueeze(-1)
        elif self.config.pooling == 'cls':
            pool_output = plm_outputs[1]
        pooled_output = self.dropout(pool_output)
        
        return self.output(pooled_output)


class PLMLSTMRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.lstm_hidden_size*2, config.num_labels)
        self.lstm = nn.LSTM(input_size=self.plm.config.hidden_size, 
                            hidden_size=config.lstm_hidden_size, 
                            num_layers=2, 
                            batch_first=True,
                            bidirectional=True)
        self.config = config
    
    def forward(self, input_ids, attention_mask):
        output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_state)  # [batch_size, max_length, 512]
        out = lstm_out.mean(dim=1)  # [batch_size, 512]
        pooled_output = self.dropout(out)
        final_output = self.output(pooled_output)
        return final_output

class PLMMultiTaskRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.dis_output = nn.Linear(self.plm.config.hidden_size, config.num_labels)
        self.emp_output = nn.Linear(self.plm.config.hidden_size, config.num_labels)
        self.config = config
    
    def forward(self, input_ids, attention_mask):
        plm_outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = plm_outputs.last_hidden_state
        last_hidden_state_sum = torch.sum(last_hidden_state*attention_mask.unsqueeze(-1), 1)
        mean_outputs = last_hidden_state_sum / attention_mask.sum(-1).unsqueeze(-1)
        cls_outputs = plm_outputs[1]
        
        mean_outputs = self.dropout(mean_outputs)
        cls_outputs = self.dropout(cls_outputs)
        
        return self.dis_output(mean_outputs), self.emp_output(cls_outputs)


@deprecated(reason="Neutral分类chatgpt top3方案无效果")
class PLMNeutralClassifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(self.plm.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(output[1])
        return self.output(pooled_output)


@deprecated(reason="multiheadattention调参无效果")
class PLMMultiHeadAttentionClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(self.plm.config.hidden_size, config.num_labels)
        self.attention = nn.MultiheadAttention(self.plm.config.hidden_size, config.num_heads, batch_first=True)
        self.config = config
    
    def forward(self, input_ids, attention_mask):
        plm_outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)        
        last_hidden_state = plm_outputs.last_hidden_state
        last_hidden_state = last_hidden_state.mean(dim=1)
        final_outputs = self.dropout(last_hidden_state)
        # cls_hidden_state = plm_outputs[1] # bsz x hsz
        # attention_outputs = self.attention(last_hidden_state, last_hidden_state, last_hidden_state)
        # attention_output = attention_outputs[0].mean(dim=1)
        # final_outputs = self.dropout(torch.cat((cls_hidden_state, attention_output), dim=-1))
        return self.output(final_outputs)


@deprecated(reason="模型过于复杂，数据集小难收敛")
class PLMLSTMMultiHeadAttentionClassifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(2*self.plm.config.hidden_size, config.num_labels)
        self.lstm = nn.LSTM(input_size=self.plm.config.hidden_size, 
                            hidden_size=int(self.plm.config.hidden_size/2), 
                            num_layers=2, 
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)

        self.attention = nn.MultiheadAttention(self.plm.config.hidden_size, config.num_heads, batch_first=True)
        self.config = config
    
    def forward(self, input_ids, attention_mask):
        plm_outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask) 
        last_hidden_state = plm_outputs.last_hidden_state # bsz x seq_len x hsz
        # batch_size, _, _ = last_hidden_state.shape
        # emotion_label_mask_expanded = emotion_label_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        # emotion_hidden_state = torch.masked_select(last_hidden_state, emotion_label_mask_expanded)
        # emotion_hidden_state = emotion_hidden_state.reshape(batch_size, self.config.num_labels, -1) # bsz x num_labels x hsz
        cls_hidden_state = plm_outputs[1] # bsz x hsz

        sentence_hidden_state = last_hidden_state[:, 1:,:] # 1+8+2
        sentence_hidden_state, _ = self.lstm(sentence_hidden_state)  # [batch_size, max_length, 512]

        attention_outputs = self.attention(sentence_hidden_state, sentence_hidden_state, sentence_hidden_state) # bsz x num_labels x hsz
        # cls_hidden_state_expanded = cls_hidden_state.unsqueeze(1).expand(-1, 8, -1)
        final_outputs = torch.cat((cls_hidden_state, attention_outputs[0].mean(1)), dim=-1)
        pooled_output = self.dropout(final_outputs)
        return self.output(pooled_output).squeeze(-1)


@deprecated(reason="投票法暂时废弃，后面有时间再探索")
class PLMVoteRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.plm = AutoModel.from_pretrained(config.plm_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(self.plm.config.hidden_size, config.num_labels)
        self.vote_layers = nn.ModuleList(nn.Linear(self.plm.config.hidden_size, 7) for i in range(config.voter_num))
        self.config = config
    
    def forward(self, input_ids, attention_mask):
        output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        
            
        last_hidden_state = output.last_hidden_state
        vote_output = last_hidden_state.narrow(dim=1, start=1, length=self.config.voter_num) # batch_size x config.voter_num x 768
        all_votes = []
        for i, l in enumerate(self.vote_layers):
            all_votes.append(self.dropout(l(vote_output[:, i, :].squeeze(1))))
        all_votes = torch.cat(all_votes, dim=-1)
        all_votes = all_votes.reshape(-1, self.config.voter_num, 7)
        logits = F.gumbel_softmax(all_votes, tau=1, hard=True) # gumbel_softmax + straight-through
        # TODO:待实现纯straight-through
        logits = torch.sum(logits, dim=1)
        scores = torch.tensor([1,2,3,4,5,6,7], device = self.config.device)
        logits = torch.sum(torch.mul(logits, scores), dim=1) / 7
        return logits.unsqueeze(1)
