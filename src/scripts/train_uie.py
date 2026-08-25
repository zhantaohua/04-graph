from pprint import pprint
import sys
from pathlib import Path
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.path.insert(0, str(PROJECT_ROOT / "external_lib"/"uie_pytorch"))
from configs import config
from uie_predictor import UIEPredictor
schema = ["商品", "颜色","运行内存"]
ie = UIEPredictor( model="uie-base", schema=schema,
task_path=config.CHECKPOINT_DIR/"uie"/"model_best")
pprint(ie( "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 8GB+128GB 冷杉绿 5G手机"))