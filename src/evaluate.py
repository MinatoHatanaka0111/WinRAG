import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os

# 先ほど作成した各クラスをインポート
from retriever import LogRetriever
from llm_engine import LLMInferenceEngine
from orchestrator import LogAnomalyOrchestrator

def extract_label(response):
    """LLMの回答テキストからラベルを抽出する補助関数"""
    response_lower = response.lower()
    if 'abnormal' in response_lower:
        return 'Abnormal'
    elif 'normal' in response_lower:
        return 'Normal'
    return 'Unknown'

def main(test_csv_path, num_samples=100):
    # 1. コンポーネントの初期化
    retriever = LogRetriever(
        index_path="./data/output/cicids_normal.index",
        messages_path="./data/output/sampled_text.csv"
    )
    
    llm_engine = LLMInferenceEngine(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    
    # オーケストレーターの構築
    system = LogAnomalyOrchestrator(retriever, llm_engine)

    # 2. テストデータの読み込み
    print(f"Loading test data from {test_csv_path}...")
    df_test = pd.read_csv(test_csv_path)
    
    # 指定したサンプル数だけランダムに抽出（全件やる場合は sample を外す）
    df_test_sample = df_test.sample(n=num_samples, random_state=42)
    
    results = []
    
    # 3. 推論ループ
    print(f"Running evaluation on {num_samples} samples...")
    for _, row in tqdm(df_test_sample.iterrows(), total=num_samples):
        # 正解ラベルの変換 ('-' なら Normal, それ以外は Abnormal)
        true_label = 'Normal' if row['Label'] == 'Benign' else 'Abnormal'
        
        # システム（オーケストレーター）による検知
        # verbose=False にして判定のみを高速に行う
        raw_response = system.detect_anomaly(row['text'], verbose=False)
        pred_label = extract_label(raw_response)
        
        results.append({
            'message': row['text'],
            'true_label': true_label,
            'pred_label': pred_label,
            'raw_response': raw_response
        })

    # 4. 結果の集計と保存
    df_results = pd.DataFrame(results)
    os.makedirs('output', exist_ok=True)
    df_results.to_csv('./data/output/test_results.csv', index=False)

    # 5. 指標の表示
    print("\n--- Evaluation Report ---")
    y_true = df_results['true_label']
    y_pred = df_results['pred_label']
    
    print(classification_report(y_true, y_pred, labels=['Abnormal', 'Normal']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=['Normal', 'Abnormal']))

if __name__ == "__main__":
    # BGLのテストデータパスを指定して実行
    main('./data/processed/cicids/test_rag_100.csv', num_samples=100)