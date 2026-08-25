import torch
from transformers import AutoTokenizer
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import config
from models.spell_check_bert import SpellCheckBert


class SpellCheckBertPredictor:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def predict(self, inputs: list[str] | str):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]

        # 处理输入数据
        encoded = self.tokenizer(
            inputs,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        predictions = outputs['predictions']

        batch_result: list[str] = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        batch_result = [result.replace(' ', '') for result in batch_result]

        if is_str:
            return batch_result[0]
        return batch_result


if __name__ == '__main__':
    model = SpellCheckBert()
    model.load_state_dict(
        torch.load(
            config.CHECKPOINT_DIR / 'spell_check_bert' / 'best.pt',
            map_location=torch.device('cpu')
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.PRE_TRAINED_DIR / 'bert-base-chinese'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SpellCheckBertPredictor(model, tokenizer, device)

    print(predictor.predict([
        '再加上在工作的地方有机会见面别的人，也可以学习新的文，最后经验越来越多。职业女生会增加她们的新智。',
        '我喜你。'
    ]))