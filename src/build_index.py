import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import faiss
import os

def build_vector_db(input_csv, output_dir, n_clusters=5, samples_per_cluster=10000):
    # 1. データの読み込み
    df = pd.read_csv(input_csv)
    messages = df['message'].astype(str).tolist()
    
    # 2. BERTによるベクトル化 (論文のEmbeddingモデルの代用)
    print("Encoding log messages using BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2') # 軽量で高性能なモデル
    embeddings = model.encode(messages, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. k-means クラスタリング
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    df['cluster'] = cluster_labels
    
    # 4. 各クラスターから均等にサンプリング
    sampled_indices = []
    for i in range(n_clusters):
        cluster_indices = df[df['cluster'] == i].index.tolist()
        # クラスター内のデータ数が指定より少ない場合は全部、多い場合はサンプリング
        n_samples = min(len(cluster_indices), samples_per_cluster)
        sampled_indices.extend(np.random.choice(cluster_indices, n_samples, replace=False))
    
    sampled_embeddings = embeddings[sampled_indices]
    sampled_texts = [messages[i] for i in sampled_indices]
    
    # 5. FAISS インデックスの構築と保存
    print("Building FAISS index...")
    dimension = sampled_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(sampled_embeddings.astype('float32'))
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "cicids_normal.index"))
    
    # 後でLLMに渡すコンテキスト用にテキストデータも保存
    pd.DataFrame({'message': sampled_texts}).to_csv(os.path.join(output_dir, "sampled_messages.csv"), index=False)
    
    print(f"Index built with {len(sampled_texts)} entries.")

# 実行例
build_vector_db('data/processed/cicids/cicids_normal_all.csv', 'data/output/')