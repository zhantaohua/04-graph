import sys
from pathlib import Path
from neo4j import GraphDatabase

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external_lib" / "uie_pytorch"))

from uie_predictor import UIEPredictor  # type: ignore
from configs import config
from runner.Predictor import IntentClassifyBertPredictor
from agent.spell_check_agent import SpellCheckAgent
import logging

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.intent_classify_predictor = self.init_intent_classfiy_predictor()
        self.spell_check_agent = self.init_spell_check_agent()
        self.uie_predictor = self.init_uie_predictor()
        self.neo4j_driver = GraphDatabase.driver(
            uri=config.NEO4J_CONFIG["uri"],
            auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]),
        )

    @staticmethod
    def init_intent_classfiy_predictor():
        model_path = config.CHECKPOINT_DIR / "intent_classify" / "best_model"
        return IntentClassifyBertPredictor(model_path)

    @staticmethod
    def init_spell_check_agent():
        return SpellCheckAgent(model_name="gpt-4o", temperature=0.2)

    @staticmethod
    def init_uie_predictor():
        return UIEPredictor(
            model='uie-base', schema=[],
            task_path=config.CHECKPOINT_DIR / "uie" / 'model_best',
        )

    def extract_entity(self, question: str, schema):
        """UIE 实体抽取，并把结果简化成 {槽位名: [文本列表]}"""
        self.uie_predictor.set_schema(schema)
        result = self.uie_predictor(question)[0]
        # 原始: {'商品': [{'text':'小米12S Ultra','probability':0.99,'start':0,'end':11}, ...]}
        # 简化: {'商品': ['小米12S Ultra', ...]}
        for key in result.keys():
            result[key] = [item["text"] for item in result[key]]
        return result

    # ===================== 单轮聊天主流程 =====================
    def chat(self, question: str) -> str:
        """
        单轮聊天入口。当前重点实现意图：【查询某商品的所有单品】
        示例：'小米12S Ultre 都有哪些版本' -> 列出该 SPU 下所有 SKU
        """

        # ---------- Step 1: 拼写纠错 ----------
        # 先把错别字纠正，让后续意图识别/实体抽取拿到干净文本
        corrected = self.spell_check_agent.correct(question)
        question = corrected.corrected_text
        logger.info(f"Step1 拼写纠错: '{question}'")

        # ---------- Step 2: 意图识别 ----------
        intent_result = self.intent_classify_predictor.predict_intent(question)
        intent = intent_result.get("predicted_intent", "")
        logger.info(f"Step2 意图识别: '{intent}'")

        # ---------- Step 3~5: 按意图分发处理 ----------
        match intent:

            # ===== 核心功能：查询某商品的所有单品 =====
            case "查询某商品的所有单品":
                return self._query_all_skus(question)

            # ===== 其它意图：暂为占位（本次不实现）=====
            case "查询某商品的某个属性的属性值":
                logger.info("命中'查询属性'意图，但该功能尚未实现")
                return "查询商品属性的功能正在开发中，敬请期待～"

        # ---------- 兜底：无法处理的意图 ----------
        logger.info(f"未匹配到可处理的意图: '{intent}'")
        return f"意图为：{intent}，该问题无法回答，请换种方式试一试！"

    # ===================== 查询某商品的所有单品（核心实现）=====================
    def _query_all_skus(self, question: str) -> str:
        """
        实现【查询某商品的所有单品】的完整子流程：
          Step3 实体抽取(商品) -> Step4 Neo4j 查 SPU 下属 SKU -> Step5 组装回答
        """
        # ---------- Step 3: 实体抽取（槽位填充：商品）----------
        schema = ["商品"]
        entity_result = self.extract_entity(question, schema)
        # entity_result = {"商品": ["小米12S Ultra"]}
        logger.info(f"Step3 实体抽取: {entity_result}")

        # 防御：没抽到任何商品
        if not entity_result.get("商品"):
            return "我没有从您的问题中识别出具体的商品名称，请换个说法再试试～"

        spu_names = entity_result["商品"]           # list，如 ["小米12S Ultra"]
        spu_name_str = "、".join(spu_names)         # 用于展示，如 "小米12S Ultra"

        # ---------- Step 4: 知识图谱查询（SPU -> 所有 SKU）----------
        cypher = """
            MATCH (spu:SPU)<-[:Belong]-(s:SKU)
            WHERE spu.spu_name IN $spu_names
            RETURN s.sku_name AS sku_name
        """
        slot = {"spu_names": spu_names}
        records, _, _ = self.neo4j_driver.execute_query(cypher, slot)
        logger.info(f"Step4 图谱查询: 命中 {len(records)} 条 SKU")

        # ---------- Step 5: 组装自然语言回答 ----------
        if len(records) > 0:
            sku_list = "\n".join([f"- {record['sku_name']}" for record in records])
            response = f"{spu_name_str} 的所有单品有：\n{sku_list}"
            return response
        else:
            # 查无此商品时的友好兜底（而不是掉到"无法回答"）
            logger.warning(f"知识图谱中未找到商品【{spu_name_str}】的单品")
            return f"抱歉，知识图谱中暂时没有【{spu_name_str}】的相关单品，请确认商品名称是否正确～"