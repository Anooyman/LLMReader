
from langchain.text_splitter import TextSplitter
import tiktoken
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StrictOverlapSplitter(TextSplitter):
    """
    严格控制相邻文本块之间重叠段落数量的文本分割器
    确保相邻块之间只重叠指定数量的段落，不会出现额外重复
    """
    def __init__(
            self, 
            overlap: int = 0, 
            token_threshold: int = 1000, 
            model: str = "gpt-4o",
            delimiter: str = '\n\n',  # 新增切分符参数
            split_count = 10,
            **kwargs):

        super().__init__(** kwargs)
        self.overlap = max(0, overlap)  # 重叠的段落数量
        self.token_threshold = max(1, token_threshold)  # token阈值
        self.model = model
        self.delimiter = delimiter  # 存储切分符
        self.split_count = split_count
        
        # 初始化token编码器
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.info(f"模型 {model} 未找到，使用默认编码o200k_base")
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.encoding.encode(text))

    def split_text(self, text: str) -> list[str]:
        """
        分割文本，确保相邻块之间严格只重叠指定数量的段落
        """
        # 使用指定的分隔符分割并过滤空段落
        paragraphs = [p for p in text.split(self.delimiter) if p.strip() != '']
        total_paragraphs = len(paragraphs)
        
        if total_paragraphs <= 1:
            return paragraphs
            
        final_chunks = []
        current_start = 0
        
        while current_start < total_paragraphs:
            # 从当前位置开始构建块
            current_end = current_start
            
            # 不断添加段落，直到达到token阈值或用尽所有段落
            while current_end < total_paragraphs:
                # 尝试添加下一个段落
                next_end = current_end + 1
                current_text = self.delimiter.join(paragraphs[current_start:next_end])
                
                # 检查是否达到或超过token阈值
                if self.count_tokens(current_text) >= self.token_threshold:
                    break  # 如果超过阈值，不添加这个段落
                
                current_end = next_end
            
            # 确保至少包含一个段落
            if current_end == current_start:
                current_end = min(current_start + 1, total_paragraphs)
                # 检查单个段落是否超过阈值
                current_text = self.delimiter.join(paragraphs[current_start:current_end])
                if self.count_tokens(current_text) > self.token_threshold:
                    logger.info(f"警告: 单个段落的token数({self.count_tokens(current_text)})超过阈值({self.token_threshold})")
            
            # 添加当前块
            current_chunk = self.delimiter.join(paragraphs[current_start:current_end])
            # 修改为字典格式，添加页码信息
            final_chunks.append({
                "data": current_chunk,
                "page": str(len(final_chunks) + 1)  # 页码从1开始计数
            })
            
            # 计算下一个块的起始位置，确保只重叠指定数量的段落
            # 下一个块的起始 = 当前块的结束 - 重叠数量
            next_start = current_end - self.overlap
            
            # 确保下一个块在当前块之后，避免无限循环
            if next_start <= current_start:
                next_start = current_start + 1
                
            current_start = next_start
            
            # 防止无限循环
            if current_start >= total_paragraphs:
                break
        
        # 处理最后可能出现的重复块（如果最后一个块长度小于重叠数量）
        if len(final_chunks) > 1 and self.overlap > 0:
            last_chunk = final_chunks[-1]["data"]
            prev_chunk = final_chunks[-2]["data"]
            if last_chunk in prev_chunk:
                final_chunks.pop()

        return final_chunks

    def split_into_chunks(self, data_list):
        """
        将数据列表按指定大小分块

        Args:
            data_list (list): 需要分块的数据列表

        Returns:
            list: 包含分块后子列表的列表

        Example:
            >>> splitter = StrictOverlapSplitter(split_count=2)
            >>> splitter.split_into_chunks([1,2,3,4,5])
            [[1,2], [3,4], [5]]
        """
        return [data_list[i:i + self.split_count] for i in range(0, len(data_list), self.split_count)]