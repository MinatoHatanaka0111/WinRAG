import pandas as pd
import os

def prepare_bgl_dataset(raw_path, output_dir):
    # 保存先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading BGL dataset... (This may take a while)")
    
    # BGLログの読み込み
    # 論文によれば、最初の列がラベル（'-'は正常、それ以外は異常）
    data = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            label = parts[0]
            # ログメッセージ本体（5列目以降がメッセージ内容）
            message = " ".join(parts[8:])
            data.append([label, message])
            
    df = pd.DataFrame(data, columns=['label', 'message'])
    
    # 1. 正常ログのみを抽出 (DB構築用)
    normal_df = df[df['label'] == '-']
    normal_df.to_csv(os.path.join(output_dir, 'bgl_normal_all.csv'), index=False)
    
    # 2. テストデータの作成 (論文では全体の20%程度を評価に使用)
    # ここでは正常・異常を混ぜた評価セットを作成
    test_df = df.sample(frac=0.2, random_state=42)
    test_df.to_csv(os.path.join(output_dir, 'bgl_test.csv'), index=False)
    
    print(f"Done! Saved to {output_dir}")
    print(f"Normal logs: {len(normal_df)}, Test samples: {len(test_df)}")

# 実行例
prepare_bgl_dataset('data/raw/BGL.log', 'data/processed')