#################### 处理完后可以删除 ###########################
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
###############################################################

from datasets import load_dataset
from transformers import AutoTokenizer
from configs import config


def process_data(model: str, save_path: Path) -> None:
    """
    数据处理：加载原始拼写纠错数据 → 划分数据集 → BERT编码 → 保存

    Args:
        model:     预训练模型名称或路径（如 bert-base-chinese）
        save_path: 处理后数据集的保存路径
    """
    # ==================== 1. 加载原始数据 ====================
    dataset = load_dataset(
        'csv',
        data_files=str(config.DATA_DIR / 'spell_check' / 'raw' / 'data.txt'),
        delimiter=' ',
        header=None,
        column_names=['text', 'label']
    )['train']

    # ==================== 2. 划分数据集 ====================
    # 80% train, 10% valid, 10% test
    dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_dict['valid'], dataset_dict['test'] = (
        dataset_dict['test'].train_test_split(test_size=0.5).values()
    )
    print(dataset_dict)

    # ==================== 3. 数据编码 ====================
    tokenizer = AutoTokenizer.from_pretrained(model)

    def map_func(batch: dict) -> dict:
        """对 batch 中的 text 和 label 分别进行 tokenize"""
        # 编码输入文本（错误句子）
        encoded_text = tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=64
        )
        # 编码标签文本（正确句子）
        encoded_label = tokenizer(
            batch['label'],
            truncation=True,
            padding='max_length',
            max_length=64
        )
        return {
            'input_ids':      encoded_text['input_ids'],
            'attention_mask': encoded_text['attention_mask'],
            'labels':         encoded_label['input_ids'],
        }

    dataset_dict = dataset_dict.map(
        map_func,
        batched=True,
        remove_columns=['text', 'label']
    )

    # ==================== 4. 保存数据集 ====================
    dataset_dict.save_to_disk(save_path)
    print(f"✅ 数据集已保存至: {save_path}")


if __name__ == '__main__':
    model_name = config.PRE_TRAINED_DIR / 'bert-base-chinese'
    output_path = config.DATA_DIR / 'spell_check' / 'processed' / 'bert'
    process_data(model=model_name, save_path=output_path)