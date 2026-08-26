# 将mysql业务数据库中的数据同步到neo4j图谱中
import pymysql
from pymysql.cursors import DictCursor
from neo4j import GraphDatabase

import sys
from pathlib import Path
# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import config

def read_sku_base_info(cursor):
    cursor.execute("""
        select 
                ski.id sku_id, 
                ski.sku_name,
                spi.spu_name,
                bc3.name category3_name, 
                bc2.name as category2_name, 
                bc1.name as category1_name,
                bt.tm_name as trademark_name 
        from sku_info ski 
                left join spu_info spi on ski.spu_id = spi.id 
                left join base_category3 bc3 on spi.category3_id = bc3.id 
                left join base_category2 bc2 on bc3.category2_id=bc2.id 
                left join base_category1 bc1 on bc2.category1_id=bc1.id 
                left join base_trademark bt on spi.tm_id = bt.id
    """)
    return cursor.fetchall()

def read_sku_attr_info(cursor):
    cursor.execute("""
        select sku_id, attr_name, value_name attr_value from sku_attr_value 
        union all 
        select sku_id, sale_attr_name, sale_attr_value_name from sku_sale_attr_value
    """)
    return cursor.fetchall()

def write_sku_base_info(driver, sku_base_info):
    """
    [
        {
            'sku_id': 1, 
            'sku_name': '小米12S Ultra 骁龙8+128GB 冷杉绿 5G手机', 
            'spu_name': '小米12S Ultra', 
            'category3_name': '手机', 
            'category2_name': '手机通讯', 
            'category1_name': '手机', 
            'trademark_name': 'Redmi'
        },
        ......
    """
    for sku in sku_base_info:
        driver.execute_query("""
            MERGE (sku:SKU{sku_id:$sku_id,sku_name:$sku_name})
            MERGE (spu:SPU{spu_name:$spu_name})
            MERGE (cate3:Category3{category3_name:$category3_name})
            MERGE (cate2:Category2{category2_name:$category2_name})
            MERGE (cate1:Category1{category1_name:$category1_name})
            MERGE (tm:Trademark{trademark_name:$trademark_name})
            MERGE (sku)-[:Belong]->(spu)
            MERGE (spu)-[:Belong]->(cate3)
            MERGE (cate3)-[:Belong]->(cate2)
            MERGE (cate2)-[:Belong]->(cate1)
            MERGE (spu)-[:Belong]->(tm)
            """, 
            parameters_=sku)
    print("1. 商品信息写入成功！")
def write_sku_attr_info(driver, sku_attr_info):
    """
    [
        {
            'sku_id': 1, 
            'attr_name': '手机一级1', 
            'attr_value': '安卓手机'
        }, {
            'sku_id': 1, 
            'attr_name': '二级手机2', 
            'attr_value': '小米'
        },
        ......
    
    """
    for attr in sku_attr_info:
        driver.execute_query("""
            MATCH (sku:SKU {sku_id:$sku_id})
            MERGE (attr:Attr {attr_name:$attr_name, attr_value:$attr_value})
            MERGE (sku)-[:Have]->(attr)
        """, parameters_=attr)
    print("2. 属性信息写入成功！")

if __name__ == '__main__':
    with pymysql.connect(**config.MYSQL_CONFIG) as conn:
        with conn.cursor(cursor=DictCursor) as cursor:
            # 1.读取产品信息
            # 2.读取属性信息
            sku_base_info = read_sku_base_info(cursor)
            sku_attr_info = read_sku_attr_info(cursor)

    with GraphDatabase.driver(uri=config.NEO4J_CONFIG['uri'], auth=(config.NEO4J_CONFIG['user'], config.NEO4J_CONFIG['password'])) as driver:
        # 3.写产品信息到neo4j
        # 4.写属性信息
        write_sku_base_info(driver, sku_base_info)
        write_sku_attr_info(driver, sku_attr_info)
        
    print("3. 数据迁移完成！")