import os
import json
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

import sys
from pathlib import Path
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import config

# ==================== 配置参数 ====================
CONFIG = {
    'data_path': config.DATA_DIR / 'intent_classify/data.csv', # 数据文件
    'other_data_path': config.DATA_DIR / 'intent_classify/other_data.csv',
    'model_name': config.PRE_TRAINED_DIR / 'bert-base-chinese',# 本地模型路径
    'output_dir': config.CHECKPOINT_DIR / 'intent_classify', # 输出目录
    'max_length': config.MAX_LENGTH,
    'batch_size': config.BATCH_SIZE,
    'epochs': config.EPOCHS, #仅用于快速跑通流程，实际建议 5~10 配合早停。
    'learning_rate': config.LEARNING_RATE,
    'test_size': config.TEST_SIZE,
    'random_state': config.RANDOM_STATE,
}

def check_local_model(path):
    """校验本地模型目录完整性"""
    required = {'config.json', 'vocab.txt'}
    weight_files = {'pytorch_model.bin', 'model.safetensors'}
    
    if not os.path.isdir(path):
        raise NotADirectoryError(f"本地模型目录不存在: {path}")
    
    files = set(os.listdir(path))
    missing = required - files #集合（set）的差集运算,找出在 required 中，但不在 files 中的元素。
    if missing:
        raise FileNotFoundError(f"本地模型缺少必要文件: {missing}")
    if not files.intersection(weight_files):
        raise FileNotFoundError("未找到模型权重文件 (pytorch_model.bin 或 model.safetensors)")
        
    print(f"地模型校验通过: {path}")

# ==================== 1. 数据加载与预处理 ====================
def load_and_preprocess_data(config, other_data_path=None):
    df = pd.read_csv(config['data_path'], encoding='utf-8')
    
     # 拓展：加载并合并 Other 数据
    if other_data_path and os.path.exists(other_data_path):
        df_other = pd.read_csv(other_data_path, encoding='utf-8')
        df_other['intent'] = 'other'  # 强制统一标签
        df = pd.concat([df, df_other], ignore_index=True)
        print(f"已注入 Other 类别数据: {len(df_other)} 条")
    
    required_cols = ['text', 'intent']        
    df = df.dropna(subset=required_cols).reset_index(drop=True)#删除 text 或 intent 任一为空值（NaN/空字符串等）的行。重置DataFrame索引，使其从 0 开始连续，便于后续切片或调试。
    print(f"有效数据: {len(df)} 条")
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['intent']) #使用 sklearn 的 LabelEncoder 将字符串类型的意图名（如 "查属性"、"咨询价格"）映射为连续整数 0, 1, 2...。
    
    label_mapping = {int(i): str(l) for i, l in enumerate(label_encoder.classes_)} #label_encoder.classes_ 按字母顺序存储了所有原始意图名。这里构建一个 {数字: 原始意图名} 的字典
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2) #ensure_ascii=False 保证中文字符正常写入，indent=2 让JSON文件易读。
        
    train_df, test_df = train_test_split(
        df, test_size=config['test_size'], random_state=config['random_state'],
        stratify=df['label'] if len(df['label'].unique()) > 1 else None #是防御性写法：如果数据只有一种类别，stratify 会报错，此处安全降级为不分组。
    )
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")
    return train_df, test_df, label_encoder, label_mapping

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
    
# ==================== 3. 评估指标 ====================
def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0)
    }    
    
def train(config):
    print("开始意图识别模型微调。。。。")
    check_local_model(config['model_name'])
    
    train_df, test_df, label_encoder, label_mapping = load_and_preprocess_data(config,other_data_path=config.get("other_data_path"))

    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    train_dataset = IntentDataset(train_df, tokenizer, max_length=config['max_length'])
    test_dataset = IntentDataset(test_df, tokenizer, max_length=config['max_length'])
    
    model = BertForSequenceClassification.from_pretrained(
        config['model_name'], 
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification"
    )
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=0.01, # 权重衰减 
        warmup_ratio=0.1, # 预热
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True, # 训练结束后自动将内存中的模型替换为验证指标最好的模型
        metric_for_best_model='f1_macro',
        greater_is_better=True, #f1 acc
        logging_dir=os.path.join(config['output_dir'], 'logs'),
        logging_steps=10,
        fp16=torch.cuda.is_available(), # 混合精度训练
        dataloader_num_workers=0, # 配合 pin_memory=True 提速
        seed=config['random_state']
    )  
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("\n训练开始...")
    train_result = trainer.train()
    print(f"训练完成: {train_result.metrics}")
    
    
    print("\n测试集评估:")
    eval_result = trainer.evaluate()
    for k, v in eval_result.items(): print(f" {k}: {v:.4f}")
    
    best_model_path = os.path.join(config['output_dir'], 'best_model')
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\n最佳模型已保存: {best_model_path}")

if __name__ == '__main__':
    train(CONFIG)