import logging
from typing import List, Dict, Optional

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from src.core.llm.client import LLMBase

logging.basicConfig(
    level=logging.INFO,  # 可根据需要改为 DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class VectorDBClient(LLMBase):
    """向量数据库客户端基类"""
    def __init__(self, db_path: str, provider: str = 'azure'):
        super().__init__(provider)
        self.db_path = db_path

    def build_vector_db(self, content_docs: List[Document]) -> FAISS:
        """
        构建向量数据库。
        Build a vector database from document list.
        Args:
            content_docs (List[Document]): 文档列表。
        Returns:
            FAISS: 构建的向量数据库对象。
        """
        logger.info(f"开始构建向量数据库，文档数: {len(content_docs)}")
        vector_db = FAISS.from_documents(content_docs, self.embedding_model)
        vector_db.save_local(self.db_path)
        logger.info(f"向量数据库已保存到: {self.db_path}")
        return vector_db 
 
    def load_vector_db(self) -> FAISS:
        """
        加载本地向量数据库。
        Load local vector database.
        Args:
            vector_db_path: 向量数据库路径。
        Returns:
            FAISS: 加载的向量数据库对象。
        """
        logger.info(f"加载本地向量数据库: {self.db_path}")
        return FAISS.load_local(self.db_path, self.embedding_model, allow_dangerous_deserialization=True)

