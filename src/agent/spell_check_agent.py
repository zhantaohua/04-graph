from langchain.chat_models import init_chat_model
from langchain.agents import create_agent 
from agent.models import CorrectionResult, ErrorDetail # type: ignore
from langchain.tools import tool
from typing import List
import re
import dotenv
import logging
from rouge_score import rouge_scorer

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #filename="app.log", # 写入文件
    #filemode="a", # 追加模式
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

class SpellCheckAgent:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = init_chat_model(
            model=model_name, 
            temperature=temperature
        )
        self.tools = self._create_tools()
        self.agent = self._create_correction_agent()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def _create_tools(self)->List:    
        #专有名词验证器
        @tool
        def query_dictionary(word: str) -> str:
            """
                    查询中文词典获取词语的正确用法、近义词和常见搭配。
                    适用于验证疑似错误词汇是否为真实存在的词语。
                    """
            # 模拟词典查询（实际应用中可对接权威词典 API）
            dictionary = {
                "万豪": ("品牌名", "万豪国际酒店集团，非房企名称"),
                "万好万家": ("企业名", "浙江万好万家文化股份有限公司"),
                "东日": ("企业名", "浙江东日股份有限公司"),
                "冬日": ("普通词汇", "冬季的白天，非企业名称")
            }
            result = dictionary.get(word, ("未知词汇", "建议结合上下文判断"))
            return f"词条: {word}\n类型: {result[0]}\n说明: {result[1]}"

        #高频错误快速拦截器
        @tool
        def check_common_errors(text: str) -> str:
            """
                    检查文本中常见的中文输入错误模式（如拼音相似错误、形近字错误）。
                    """
            # 常见错误模式库：硬编码模式库（维护成本高）
            error_patterns = [
                (r"始曰苯", "使日本", "拼音输入错误"),
                (r"万豪万家", "万好万家", "企业名称混淆"),
                (r"冬日", "东日", "形近字错误（房企名称）"),
                (r"的\b", "地|得", "的/地/得混用（需结合语法判断）"),
                (r"在\b.*?在\b", "", "重复使用'在'字")
            ]
            # 改进方案：外部配置化
            # import json
            # with open("error_patterns.json") as f:
            #     error_patterns = json.load(f)  # 支持热更新
            
            matches = []
            for pattern, correction, error_type in error_patterns:
                if re.search(pattern, text):
                    matches.append(f"- '{pattern}' → '{correction}' ({error_type})")
                    print("检测到潜在错误模式:\n" + "\n".join(matches))
            
            if matches:
                return "检测到潜在错误模式:\n" + "\n".join(matches)
            return "未检测到常见错误模式"
    
        return [query_dictionary, check_common_errors]
    def _create_correction_agent(self):
        # 系统提示词
        system_prompt = """你是一个专业的中文文本纠错专家，擅长识别和修正各种类型的中文错误。
            【可用工具】
            1. query_dictionary: 查询中文词典
            2. check_common_errors: 检查常见的中文输入错误模式
            
            【错误类型】
            1. 拼写错误：拼音输入错误、形近字错误
            2. 语法错误：的/地/得混用、语序错误、成分残缺
            3. 用词不当：词语搭配错误、专业术语误用
            4. 标点错误：标点缺失、误用或冗余
            5. 语义错误：逻辑矛盾、事实错误

            【输出要求】
            1. 严格按 JSON Schema 输出，不要添加额外说明
            2. 错误位置使用字符索引（从0开始）
            3. 修正理由需结合上下文解释
            4. 置信度计算：高确定性错误(0.9+)，需推断错误(0.7-0.9)，不确定(<0.7) 

            【特别注意】
            - 保留原文风格和语义，只修正明确错误
            - 专有名词（人名、地名、企业名）需谨慎修正
            - 不确定时宁可不改，置信度设为低值
            """
        agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt,
                response_format=CorrectionResult
            )
        return agent
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算 Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _calculate_confidence(self, original: str, corrected: str) -> float:
        """基于 ROUGE-L 和编辑距离计算置信度"""
        if original == corrected:
            return 1.0
        
        # ROUGE-L 相似度
        # 维度1：ROUGE-L 相似度（语义保留度）
        scores = self.rouge_scorer.score(original, corrected)
        rouge_l = scores['rougeL'].fmeasure
        
        # 编辑距离比例（越小表示改动越小，置信度越高）
        edit_distance = self._levenshtein_distance(original, corrected)
        max_len = max(len(original), len(corrected))
        # 维度2：编辑距离比例（改动幅度）
        edit_ratio = edit_distance / max_len if max_len > 0 else 1.0
        
        # 综合置信度：基础值 + ROUGE 加成 - 编辑距离惩罚
        base_confidence = 0.6
        confidence = base_confidence + 0.3 * rouge_l - 0.2 * edit_ratio
        return max(0.0, min(1.0, confidence))

    def _merge_corrections(self, chunks: List[str], results: List[CorrectionResult]) -> CorrectionResult:
            """合并分块纠错结果（修复类型问题）"""
            merged_text = ""
            merged_errors = []  # 保持为 ErrorDetail 对象列表
            offset = 0
            
            for chunk, result in zip(chunks, results):
                # 关键：offset基于原始块长度累加，而非修正后长度！
                merged_text += result.corrected_text
                
                # 修复：将字典转换回 ErrorDetail 对象
                for error in result.errors:
                    # 创建新的 ErrorDetail 对象（保持Pydantic验证）
                    adjusted_error = ErrorDetail(
                        position=error.position + offset,
                        error_span=error.error_span,
                        error_type=error.error_type,
                        correction=error.correction,
                        reason=error.reason
                    )
                    merged_errors.append(adjusted_error)
                
                offset += len(chunk)
            
            avg_confidence = sum(r.confidence for r in results) / len(results)
            explanation = f"文本被分割为{len(chunks)}个语义块分别纠错，综合置信度:{avg_confidence:.2f}"
            
            return CorrectionResult(
                original_text="".join(chunks),
                corrected_text=merged_text,
                errors=merged_errors,  # 确保是 ErrorDetail 对象列表
                confidence=avg_confidence,
                explanation=explanation
            )
    
    def _split_long_text(self, text: str, max_length: int = 300) -> List[str]:
        """将长文本智能分割为语义连贯的块"""
        # 优先按句子分割（保留标点完整性）
        sentences = re.split(r'(?<=[。！？；])', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip()) # 保存当前块
                current_chunk = sentence  # 新块从当前句开始
            else:
                current_chunk += sentence # 继续累加
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if len(chunks) > 1 else [text]
    def correct(self, text:str)->CorrectionResult:
        """
        执行纠错错误
        """
        # 1. 长块切分
        chunks = self._split_long_text(text)
        
        # 2. 逐块处理
        results = []
        for chunk in chunks:
            try:
                response = self.agent.invoke({
                    "messages": [{"role": "user", "content": chunk}]
                })
                result = response.get("structured_response")

                if result is None: #忽略无效结果
                    continue
                # 3. 计算每块的置信度
                confidence = self._calculate_confidence(chunk, result.corrected_text)
                # 模型的评分和我们的评分取最小值
                result.confidence = min(result.confidence, confidence)
                results.append(result)
                
            except Exception as e:
                # 降级处理
                logger.error(f"Error processing chunk: {e}")
                results.append(CorrectionResult(
                    corrected_text=chunk,
                    original_text=chunk,
                    confidence=0.0,
                    errors=[],
                    explanation=f"模型处理异常！{e}"
                ))
        
        # 4. 合并结果
        if len(results)>1:
            #合并多块
            merged_result = self._merge_corrections(chunks, results)
        else:
            merged_result = results[0]
            
        # 判断置信度是否过低
        if merged_result.confidence < 0.4:
            return CorrectionResult(
                corrected_text=text,
                original_text=text,
                confidence=merged_result.confidence,
                errors=[],
                explanation=f"置信度过低:{merged_result.confidence:.2f}，建议保留！"
            )
        else:
            return merged_result
        
if __name__ == "__main__":
    spell_agent = SpellCheckAgent()
    result = spell_agent.correct("我们在一起玩德很开心")
    print(result)