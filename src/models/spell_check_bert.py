import torch
from torch import nn, tensor
from transformers import AutoModel

from configs import config


class SpellCheckBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.bert.config.pad_token_id)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.linear(last_hidden_state)
        # logits.shape: [batch_size,seq_len,vocab_size]
        predictions = torch.argmax(logits, dim=-1)
        # predictions.shape: [batch_size,seq_len]

        # torch.cat([torch.full((batch_size, 1), 0),attention_mask[:,2:],torch.full((batch_size, 1), 0)])
        # predictions.shape: [batch_size,seq_len]
        predictions = predictions.masked_fill(attention_mask == 0, self.bert.config.pad_token_id)

        result = {'predictions': predictions}
        if labels is not None:
            loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            result['loss'] = loss
        return result

# ==================== 2. Dataset 类 ====================
class IntentDataset:
    def __init__(self, dataframe, tokenizer, max_length=128, text_col='text', label_col='label'):
        #.tolist() 优化：提前将 pandas Series 转为 Python 原生列表。避免每次 __getitem__ 时触发 pandas 索引开销，提升迭代速度。
        self.texts = dataframe[text_col].tolist()
        #兼容无标签场景：labels 允许为 None。这意味着同一份 Dataset 类既能用于训练（有标签），也能用于线上推理/批量预测（无标签），提高代码复用率。
        self.labels = dataframe[label_col].tolist() if label_col in dataframe.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), 
            truncation=True, 
            padding=False,
            max_length=self.max_length, 
            return_token_type_ids=True
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) #dtype=torch.long 必须项。PyTorch 分类损失函数 nn.CrossEntropyLoss 要求目标标签为 64位整型（torch.long），否则会报类型不匹配错误。
        return item