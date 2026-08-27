import torch
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification
import json
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
        inputs = self.tokenizer(inputs,
                                truncation=True,
                                padding='max_length',
                                max_length=64,
                                return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        predictions = outputs['predictions']

        batch_result: list[str] = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        batch_result = [result.replace(' ', '') for result in batch_result]
        if is_str:
            return batch_result[0]
        return batch_result


class IntentClassifyBertPredictor:
    def __init__(self, model_path):
        #self.model = BertForSequenceClassification.from_pretrained(model_path, map_location="cpu") # cpu上跑模型
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device) # type: ignore
        with open(config.CHECKPOINT_DIR / 'intent_classify' / 'label_mapping.json', 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
    def predict_intent(self, text, max_length=128,threshold=0.7):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()} #将值放到和model一样的设备上
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item() # type: ignore
            
        pred_label = self.label_mapping.get(str(pred_id), f"unknown")
        is_other = (pred_label == 'other') or (pred_label == 'unknown')
        is_low = (confidence < threshold) and not is_other
        
        if is_other: # 其他意图
            return {
                "text": text,
                "predicted_intent": "未知意图",
                "confidence": round(confidence, 4),
                "is_other": True,
                "label_id": pred_id,
                "recommend_action":"1. 转人工或触发兜底回复！"
            }
        if is_low: # 置信度过低
            return {
                "text": text,
                "predicted_intent": pred_label,
                "confidence": round(confidence, 4),
                "is_other": False,
                "label_id": pred_id,
                "recommend_action":"2. 建议人工复核！"
            }
        
        return {
            "text": text,
            "predicted_intent": pred_label,
            "confidence": round(confidence, 4),
            "is_other": False,
            "label_id": pred_id,
            "recommend_action":"3. 正常处理"
        }

if __name__ == '__main__':
    # model = SpellCheckBert()
    # model.load_state_dict(torch.load(config.CHECKPOINT_DIR / 'spell_check_bert' / 'best.pt',map_location=torch.device('cpu')))

    # tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # predictor = SpellCheckBertPredictor(model, tokenizer, device)

    # print(predictor.predict(
    #     ['我喜换你',
    #      '我喜你。'])) # bert, t5
    
    predictor = IntentClassifyBertPredictor(config.CHECKPOINT_DIR / 'intent_classify' / 'best_model')
    text = "小米手机都有哪些版本" #{'text': '小米手机都有哪些版本', 'predicted_intent': '查询某品类某品牌的所有商品', 'confidence': 0.471, 'is_other': False, 'label_id': 10, 'recommend_action': '2. 建议人工复核！'}
    #text = "小米12S Ultra 都有哪些版本" #{'text': '小米12S Ultra 都有哪些版本', 'predicted_intent': '查询某商品的所有单品', 'confidence': 0.7487, 'is_other': False, 'label_id': 13, 'recommend_action': '3. 正常处理'}
    #text = "hello world" #{'text': 'hello world', 'predicted_intent': '未知意图', 'confidence': 0.9918, 'is_other': True, 'label_id': 0, 'recommend_action': '1. 转人工或触发兜底回复！'}
    print(predictor.predict_intent(text))