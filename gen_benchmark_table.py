import pandas as pd
from pathlib import Path

df = pd.read_excel('benchmark_results.xlsx', sheet_name='Benchmark Results')
cols = ['Model_Name','Architecture','Training_Data','TaskA_PER_HighQuality','TaskA_Accuracy','TaskB_Pearson_r','TaskB_Spearman_rho','TaskC_AUC_ROC','TaskC_F1_Score','TaskC_Recall_Errors','TaskC_Precision','TaskC_Threshold','Notes']
df = df[cols]
for c in [c for c in df.columns if c.startswith('Task')]:
    df[c] = df[c].apply(lambda x: '' if pd.isna(x) else float(x))
round_map = {
    'TaskA_PER_HighQuality': 2,
    'TaskA_Accuracy': 2,
    'TaskB_Pearson_r': 4,
    'TaskB_Spearman_rho': 4,
    'TaskC_AUC_ROC': 4,
    'TaskC_F1_Score': 4,
    'TaskC_Recall_Errors': 4,
    'TaskC_Precision': 4,
    'TaskC_Threshold': 2,
}
for c, nd in round_map.items():
    df[c] = df[c].apply(lambda x: '' if x=='' else round(x, nd))

md = df.to_markdown(index=False)
Path('docs').mkdir(exist_ok=True)
Path('docs/BENCHMARK_RESULTS_TABLE.md').write_text(md, encoding='utf-8')
print('Wrote docs/BENCHMARK_RESULTS_TABLE.md')
