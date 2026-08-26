# 将商品图片的内容识别出来，并实体抽取写入图谱
import pymysql
from pymysql.cursors import DictCursor
from neo4j import GraphDatabase
import easyocr
from tqdm import tqdm

import sys
from pathlib import Path
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external_lib"/"uie_pytorch"))

from configs import config
from agent.spell_check_agent import SpellCheckAgent
from uie_predictor import UIEPredictor # type: ignore
import logging
# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename="app.log", # 写入文件
    #filemode="a", # 追加模式
)
logger = logging.getLogger(__name__)

agent = SpellCheckAgent()

"""
[
    {
        'sku_id': 36, 
        'img_url': '/data/images/36/1.jpg'
    }, {
        'sku_id': 36, 
        'img_url': '/data/images/36/2.jpg'
    },
    ......
"""
def get_sku_image_url():
    logger.info("1. 开始获取商品图片的URL...")
    with pymysql.connect(**config.MYSQL_CONFIG) as conn:
        with conn.cursor(DictCursor) as cursor:
            sql = "select sku_id,img_url from sku_image where img_url like '/data%'"
            cursor.execute(sql)
            return cursor.fetchall()
"""
{
    "sku_id":[1, 2, 3]
    "img_content":["xxxx","yyyyy","zzzz"]
}
"""  
def get_sku_image_content(images_url): 
    logger.info("2. 开始获取商品图片的内容...")
    sku_image_content = {"sku_id":[], "img_content":[]}
    reader = easyocr.Reader(['ch_sim','en'])
    #i = 0
    for image_url in tqdm(images_url, desc="图片识别中..."): 
        """
        需要识别的图片为：
        {
            'sku_id': 36, 
            'img_url': '/data/images/36/1.jpg'
        }
        """
        try:
            #i = i + 1
            image_path = image_url["img_url"][1:]
            result = reader.readtext(image_path, detail=0)
            image_content = "".join(result)  # type: ignore
        except Exception as e:
            print(f"图片识别失败：{image_url["img_url"]}")
            image_content = ""
        sku_image_content["sku_id"].append(image_url["sku_id"])
        sku_image_content["img_content"].append(image_content)

        #if i == 5:
        #    break
    return sku_image_content

"""
{
    "sku_id":[1, 2, 3]
    "img_content":["纠错之后的xxxx","yyyyy","zzzz"]
}
""" 
def correct_sku_image_content(sku_image_content):
    logger.info("3. 纠正商品图片的内容...")
    for i, content in enumerate(sku_image_content["img_content"]): 
        #print(f"正在对第{i}张图片的内容进行纠错：{content}")
        result = agent.correct(content)
        sku_image_content["img_content"][i] = result.corrected_text
        #print(f"第{i}张图片的纠正的结果为：{result.corrected_text}")
    return sku_image_content
  
"""
{
    "sku_id":[1, 2, 3]
    "desc_content":["xxxx","yyyyy","zzzz"]
}
"""

def get_sku_detail_content(): 
    logger.info("4. 获取商品详情的内容...")
    sku_desc_content = {"sku_id":[], "sku_desc":[]}
    with pymysql.connect(**config.MYSQL_CONFIG) as conn:
        with conn.cursor(DictCursor) as cursor:
            #sql = "select id sku_id, sku_desc from sku_info limit 5"
            sql = "select id sku_id, sku_desc from sku_info"
            cursor.execute(sql)
            result = cursor.fetchall() 
    
    for item in result: 
        sku_desc_content["sku_id"].append(item["sku_id"])
        sku_desc_content["sku_desc"].append(item["sku_desc"])
    return sku_desc_content

"""
输入的数据结构：
{
    'sku_id': [36,43......], 
    'sku_content': ['购买客户尊享会员服务'，'全国联保。'......],
}

使用UIE模型进行实体抽取
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
输出的数据结构：
[
    {
        "sku_id": 36,
        "attr_name": "版本", 
        "attr_value": "5G"
    },
    {
        "sku_id": 36,
        "attr_name": "颜色", 
        "attr_value": "粉色"
    },
    {
        "sku_id": 36,
        "attr_name": "品类", 
        "attr_value": "电脑"
    },
    {
        "sku_id": 43,
        "attr_name": "运行内存", 
        "attr_value": "12G"
    }
    ......
]

"""
def get_sku_entity(sku_contents):
    logger.info("5. 抽取商品详情的实体...")
    sku_ids = sku_contents["sku_id"]
    # 实体抽取的schema?品牌、价格、运行内存。。。
    ie = UIEPredictor(model='uie-base', schema=config.SCHEMA, task_path=config.CHECKPOINT_DIR/"uie" / 'model_best')
    result = ie(sku_contents["sku_content"])
    """
        [{'商品': [{'end': 11,
                'probability': np.float32(0.9976633),
                'start': 0,
                'text': '小米12S Ultra'}],
        '运行内存': [{'end': 53,
                    'probability': np.float32(0.9690239),
                    'start': 50,
                    'text': '8GB'}],
        '颜色': [{'end': 63,
                'probability': np.float32(0.99969167),
                'start': 60,
                'text': '冷杉绿'}]},
        {'商品': [{'end': 11,
                'probability': np.float32(0.99801755),
                'start': 0,
                'text': '小米12S Ultra'}],
        '运行内存': [{'end': 53,
                    'probability': np.float32(0.987109),
                    'start': 50,
                    'text': '8GB'}],
        '颜色': [{'end': 63,
                'probability': np.float32(0.99875057),
                'start': 60,
                'text': '经典黑'}]}
                
                ......]
                
        转换数据结构：
        [
            {
                "sku_id": sku_ids[0],
                "attr_name": "商品", 
                "attr_value": result[0]["商品"][0]["text"]
            },
            {
                "sku_id": sku_ids[0],
                "attr_name": "运行内存", 
                "attr_value":  result[0]["运行内存"][0]["text"]
            },
            {
                "sku_id": sku_ids[1],
                "attr_name": "商品", 
                "attr_value":  result[1]["商品"][0]["text"]
            },
            {
                "sku_id": sku_ids[1],
                "attr_name": "运行内存", 
                "attr_value":  result[1]["运行内存"][0]["text"]
            }
        ]
        
    """
    sku_entity = []
    for i, item in enumerate(result):
        for attr_name, attr_values in item.items(): 
            if attr_name in config.SCHEMA: 
                sku_entity.append({
                    "sku_id": sku_ids[i],
                    "attr_name": attr_name,
                    "attr_value": attr_values[0]["text"]
                })

    return sku_entity

"""
[
    {
        "sku_id": 36,
        "attr_name": "版本", 
        "attr_value": "5G"
    },
    {
        "sku_id": 36,
        "attr_name": "颜色", 
        "attr_value": "粉色"
    },
    ......

"""
def write_sku_attr_info(sku_entity):
    logger.info("6. 写入商品属性信息...")
    logger.info(f"正在写入{len(sku_entity)}条数据...")
    with GraphDatabase.driver(
        uri=config.NEO4J_CONFIG['uri'], 
        auth=(config.NEO4J_CONFIG['user'], config.NEO4J_CONFIG['password'])) as driver:
        for item in tqdm(sku_entity, desc="写入商品属性信息..."): 
            driver.execute_query(
                """
                MATCH (sku:SKU{sku_id:$sku_id})
                OPTIONAL MATCH (sku)-[:Have]->(attr_exists:Attr{attr_name:$attr_name, attr_value:$attr_value})
                WITH sku, attr_exists
                WHERE attr_exists  is null
                MERGE (attr:Attr{attr_name:$attr_name, attr_value:$attr_value})
                MERGE (sku)-[:Have]->(attr)
                """
            , parameters_=item)

if __name__ == '__main__':
    # 1.获取图片
    images_url = get_sku_image_url()
    #print(images_url)
    
    # 2.图片识别
    sku_image_content = get_sku_image_content(images_url)
    #print(sku_image_content)
    
    # 3.对识别结果进行拼写纠错
    sku_image_content = correct_sku_image_content(sku_image_content)
    #print(sku_image_content)
    
    # 4.读取商品详情的内容
    sku_detail_content = get_sku_detail_content()
    #print(sku_detail_content)
    
    # 5.将图片的内容和商品详情的内容合并在一起
    sku_content = {
        "sku_id":sku_image_content["sku_id"] + sku_detail_content["sku_id"], #[1,2,3] 
        "sku_content":sku_image_content["img_content"] + sku_detail_content["sku_desc"] #["xxx", "yyy", "zzz"]
    }
    #print(sku_content)
    
    # 6.实体抽取
    sku_entity = get_sku_entity(sku_content)
    #print(sku_entity)
    
    # 5.写入图谱
    write_sku_attr_info(sku_entity)
    
    logger.info("数据迁移完成！")