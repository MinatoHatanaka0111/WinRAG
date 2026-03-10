from retriever import LogRetriever
from llm_engine import LLMInferenceEngine
from orchestrator import LogAnomalyOrchestrator

def main():
    # 各コンポーネントの初期化
    retriever = LogRetriever(
        index_path="./data/output/cicids_normal.index",
        messages_path="./data/output/sampled_text.csv"
    )
    
    llm_engine = LLMInferenceEngine(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    
    # オーケストレーターの構築
    system = LogAnomalyOrchestrator(retriever, llm_engine)
    
    # 実行
    test_log = "Dst Port: 22 | Protocol: 6 | Flow Duration: 6 | Tot Fwd Pkts: 1 | Tot Bwd Pkts: 1 | TotLen Fwd Pkts: 0 | Flow Byts/s: 0 | Flow Pkts/s: 333333.33 | Init Fwd Win Byts: 241 | Idle Mean: 0"
    result = system.detect_anomaly(test_log, verbose=True)
    
    print(f"\n[Log]: {test_log}")
    print(f"[Result]:\n{result}")

if __name__ == "__main__":
    main()