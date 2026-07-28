### [ 1. 라이브러리 임포트 ] ###
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import pandas as pd
from config import PROJECT_ROOT

### [ 2. eval/results_*.csv 전부 읽어서 요약표 생성 ] ###
def build_summary():
    rows = []
    pattern = os.path.join(PROJECT_ROOT, "eval", "results_*.csv")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path).replace("results_", "").replace(".csv", "")
        df = pd.read_csv(path)
        row = {
            "preset": name,
            "bertscore_f1": df["bertscore_f1"].mean(),
            "response_time_sec": df["response_time_sec"].mean(),
        }
        if "ragas_faithfulness" in df.columns:
            row["faithfulness"] = df["ragas_faithfulness"].mean()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(PROJECT_ROOT, "eval", "comparison_summary.csv")
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    print(f"\n저장 위치: {out_path}")


if __name__ == "__main__":
    build_summary()