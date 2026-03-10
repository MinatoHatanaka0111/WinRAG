class LogAnomalyOrchestrator:
    def __init__(self, retriever, llm_engine):
        self.retriever = retriever
        self.llm_engine = llm_engine

    def _build_prompt(self, query_log, context_logs, verbose=False):
        """RAG用のプロンプトを組み立てる内部メソッド"""
        context_str = "\n".join([f"- {msg}" for msg in context_logs])
        
        if verbose:
            instruction = "Provide the label and a brief reason for your decision."
        else:
            instruction = "Provide ONLY the label ('Normal' or 'Abnormal') without any explanation."

        return f"""Identify if the following 'Target Log' is 'Normal' or 'Abnormal' based on the 'Normal Log Examples'.
{instruction}

### Normal Log Examples:
{context_str}

### Target Log:
{query_log}

### Analysis:
Label:"""

    def detect_anomaly(self, query_log, verbose=False):
        """
        1. 検索 (Retrieval)
        2. プロンプト構築
        3. 推論 (Inference)
        の一連の流れを実行する
        """
        # 1. 類似ログの取得
        relevant_logs = self.retriever.retrieve(query_log, k=3)
        
        # 2. プロンプトの組み立て
        prompt = self._build_prompt(query_log, relevant_logs, verbose=verbose)
        
        # 3. LLMで推論
        max_tokens = 100 if verbose else 10
        raw_response = self.llm_engine.generate(prompt, max_new_tokens=max_tokens)
        
        return raw_response