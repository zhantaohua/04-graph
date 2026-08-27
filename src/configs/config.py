from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

# 定义项目的目录结构
DATA_DIR = ROOT_DIR / "data"
PRE_TRAINED_DIR = ROOT_DIR / "pretrained"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
WEB_STATIC_DIR = ROOT_DIR / "src" / "web" / "static"
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'zzzz',
    'database': 'mall',
    'charset': 'utf8mb4'
}

NEO4J_CONFIG = {
    'uri': 'neo4j://localhost:7687',
    'user': 'neo4j',
    'password': '12345678'
}

SCHEMA=[
    "品类",
    "品牌",
    "商品",
    "尺码",
    "观看距离", 
    "分辨率",
    "屏幕尺寸",
    "电视类型",
    "版本",
    "颜色",
    "机身内存",
    "运行内存",
    "处理器或内存",
    "内存",
    "硬盘",
    "显卡",
    "处理器",
    "类别",
    "分类",
    "是否有机",
    "粮食调味",
    "面部护肤",
    "香水彩妆",
    "功效",
    "香调",
    "电池容量",
    "摄像头像素",
    "散热方式",
    "解锁方式"
]

MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42