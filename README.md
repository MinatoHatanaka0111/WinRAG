# WinRAG

Windowsの攻撃データをRAG（Retrieval-Augmented Generation）で解析するための実験プロジェクト。

## 概要
Windowsイベントログ（.evtx）などのセキュリティデータをLLM（大規模言語モデル）に読み込ませ、自然言語による攻撃検知やインシデント調査の自動化・高度化を検証します。

## 📊 評価レポート (Evaluation Report)

プロジェクト内の RAGLog ロジックを用いて、CIC-IDS-2018 データセット（精選 10 カラム）に対する分類性能を評価しました。

### 1. 実行環境・データ構成
- **モデル**: RAGLog (LLM + Vector Search)
- **テストデータ数**: 500 件
    - **Normal (正常)**: 307 件
    - **Abnormal (攻撃)**: 193 件
- **使用カラム**: `Dst Port`, `Protocol`, `Flow Duration` 等、厳選した 10 特徴量

### 2. 分類レポート (Classification Report)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Abnormal** | 0.66 | **0.98** | 0.79 | 193 |
| **Normal** | 0.99 | 0.68 | 0.80 | 307 |
| **Accuracy** | | | **0.80** | 500 |

### 3. 混同行列 (Confusion Matrix)

| | Predicted Normal | Predicted Abnormal |
| :--- | :---: | :---: |
| **Actual Normal** | **208** (TN) | 99 (FP) |
| **Actual Abnormal**| 3 (FN) | **190** (TP) |