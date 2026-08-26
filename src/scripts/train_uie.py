import sys
from pathlib import Path
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external_lib"/"uie_pytorch"))

from configs import config
from uie_predictor import UIEPredictor
from pprint import pprint

# schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
#ie = UIEPredictor(model='uie-base', schema=schema, task_path=config.PRE_TRAINED_DIR / 'uie_base_pytorch')
# pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌和赵国弼分别以188.25分获得金牌！")) 

"""
[{'时间': [{'end': 6,
          'probability': np.float32(0.98651797),
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': np.float32(0.84298414),
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 35,
          'probability': np.float32(0.7838276),
          'start': 32,
          'text': '赵国弼'},
         {'end': 31,
          'probability': np.float32(0.73206705),
          'start': 28,
          'text': '谷爱凌'}]}]
"""


###############################################################
#schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
#ie = UIEPredictor(model='uie-base', schema=[], task_path=config.PRE_TRAINED_DIR / 'uie_base_pytorch')
#ie.set_schema(schema)
#pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))
"""
[{'肝癌级别': [{'end': 20,
            'probability': np.float32(0.9243267),
            'start': 13,
            'text': 'II-III级'}],
  '肿瘤的个数': [{'end': 84,
             'probability': np.float32(0.75384146),
             'start': 82,
             'text': '1个'}],
  '肿瘤的大小': [{'end': 100,
             'probability': np.float32(0.8341128),
             'start': 87,
             'text': '4.2×4.0×2.8cm'}],
  '脉管内癌栓分级': [{'end': 70,
               'probability': np.float32(0.90832955),
               'start': 67,
               'text': 'M0级'}]}]
"""
# ie = UIEPredictor(model='uie-base', schema=[], task_path=config.PRE_TRAINED_DIR / 'uie_base_pytorch')
# schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']} # Define the schema for relation extraction
# ie.set_schema(schema) # Reset schema
# pprint(ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))

"""
[{'竞赛名称': [{'end': 13,
            'probability': np.float32(0.7825388),
            'relations': {'主办方': [{'end': 22,
                                   'probability': np.float32(0.8421713),
                                   'start': 14,
                                   'text': '中国中文信息学会'},
                                  {'end': 30,
                                   'probability': np.float32(0.7580802),
                                   'start': 23,
                                   'text': '中国计算机学会'}],
                          '已举办次数': [{'end': 82,
                                     'probability': np.float32(0.46712977),
                                     'start': 80,
                                     'text': '4届'}],
                          '承办方': [{'end': 55,
                                   'probability': np.float32(0.70004934),
                                   'start': 40,
                                   'text': '中国中文信息学会评测工作委员会'},
                                  {'end': 39,
                                   'probability': np.float32(0.82927024),
                                   'start': 35,
                                   'text': '百度公司'},
                                  {'end': 72,
                                   'probability': np.float32(0.61934745),
                                   'start': 56,
                                   'text': '中国计算机学会自然语言处理专委会'}]},
            'start': 0,
            'text': '2022语言与智能技术竞赛'}]}]
"""


#电商
# 请帮我推荐一款小米12s ultra的运行内存8G+128GB,颜色为冷杉绿?
# 意图识别，公式：（商品_     运行内存_    颜色_冷杉绿）   
input = ["小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 8GB+128GB 冷杉绿 5G手机", "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 8GB+256GB 经典黑 5G手机"]
ie = UIEPredictor(model='uie-base', schema=[], task_path=config.CHECKPOINT_DIR/"uie" / 'model_best')
schema = ['商品', '颜色', '运行内存']
ie.set_schema(schema)
pprint(ie(input))
"""
[{'颜色': [{'end': 63,
          'probability': np.float32(0.5169821),
          'start': 60,
          'text': '冷杉绿'}]}]
使用微调后的模型：
[{'商品': [{'end': 11,
          'probability': np.float32(0.9976633),
          'start': 0,
          'text': '小米12S Ultra'}],
  '运行内存': [{'end': 53,
            'probability': np.float32(0.969024),
            'start': 50,
            'text': '8GB'}],
  '颜色': [{'end': 63,
          'probability': np.float32(0.99969167),
          'start': 60,
          'text': '冷杉绿'}]}]
"""

#python external_lib/uie_pytorch/doccano.py  --doccano_file ./data/uie/raw/doccano.jsonl  --task_type ext  --save_dir ./data/uie/processed --splits 0.8 0.2 0


#python external_lib/uie_pytorch/finetune.py --train_path "./data/uie/processed/train.txt"  --dev_path "./data/uie/processed/dev.txt" --save_dir "./checkpoint/uie" --learning_rate 1e-5 --batch_size 16  --max_seq_len 512  --num_epochs 10  --model "./pretrained/uie_base_pytorch"  --seed 1000  --logging_steps 10  --valid_steps 100  --device "gpu" --early-stopping


#python external_lib/uie_pytorch/evaluate.py  --model_path ./checkpoint/uie/model_best --test_path ./data/uie/processed/dev.txt --batch_size 16 --max_seq_len 512