import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

class LogRetriever:
    def __init__(self, index_path, messages_path, model_name='all-MiniLM-L6-v2'):
        print("Initializing Retriever...")
        self.index = faiss.read_index(index_path)
        self.sampled_messages = pd.read_csv(messages_path)['message'].tolist()
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query_log, k=3):
        """クエリに類似したログを検索して返す"""
        query_vector = self.embedder.encode([query_log])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return [self.sampled_messages[i] for i in indices[0]]