
import pymysql
from pymysql.cursors import DictCursor
from neo4j import GraphDatabase
from tqdm import tqdm
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_users(cursor):
    cursor.execute("SELECT id AS user_id, nick_name, gender, user_level FROM user_info")
    return cursor.fetchall()

def read_view_logs(cursor):
    """从 user_view_log 读取真实浏览记录，按 user+sku 去重"""
    sql = """
        SELECT 
            user_id,
            sku_id,
            DATE_FORMAT(MAX(view_time), '%Y-%m-%d %H:%i:%s') AS last_view_time
        FROM user_view_log
        GROUP BY user_id, sku_id
    """
    cursor.execute(sql)
    return cursor.fetchall()


def write_to_neo4j(driver, users, view_logs):
    #  写入 User 节点
    logger.info(f"开始写入 {len(users)} 个 User 节点...")
    for user in tqdm(users, desc="创建 User 节点"):
        driver.execute_query("""
            MERGE (u:User {user_id: $user_id})
            SET u.nick_name = $nick_name,
                u.gender = $gender,
                u.user_level = $user_level
        """, parameters_=user)
    print("✅ User 节点写入成功！")

    # 写入 (User)-[:View]->(SKU) 关系
    logger.info(f"开始写入 {len(view_logs)} 条 View 关系...")
    skipped = 0
    for log in tqdm(view_logs, desc="创建 View 关系"):
        result = driver.execute_query("""
            MATCH (u:User {user_id: $user_id})
            MATCH (s:SKU {sku_id: $sku_id})
            MERGE (u)-[r:View]->(s)
            SET r.last_view_time = $last_view_time
            RETURN r
        """, parameters_=log)
        if not result.records:
            skipped += 1  
    
    if skipped > 0:
        logger.warning(f"有 {skipped} 条记录因找不到对应 SKU 节点被跳过")
    print("(User)-[:View]->(SKU) 关系写入成功！")


if __name__ == '__main__':
    logger.info("====== 开始构建 User-View-SKU 图谱 ======")

    with pymysql.connect(**config.MYSQL_CONFIG) as conn:
        with conn.cursor(cursor=DictCursor) as cursor:
            users = read_users(cursor)
            view_logs = read_view_logs(cursor)

    logger.info(f"数据统计: 用户 {len(users)} 人, 浏览关系(去重后) {len(view_logs)} 条")

    if not users:
        logger.error("未读取到用户数据，请检查 user_info 表")
        sys.exit(1)

    with GraphDatabase.driver(
        uri=config.NEO4J_CONFIG['uri'],
        auth=(config.NEO4J_CONFIG['user'], config.NEO4J_CONFIG['password'])
    ) as driver:
        write_to_neo4j(driver, users, view_logs)