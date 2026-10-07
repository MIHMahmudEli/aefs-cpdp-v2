from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

results = {}
for ds in ['aeeem', 'tera', 'nasa', 'relink']:
    df = pd.read_csv(ROOT / "results" / f"{ds}_extended_results.csv")
    results[ds] = df

methods = ['AEFS', 'Ablation_no2D', 'AllFeatures', 'FilterMI', 'WrapperRFE', 'BorutaSHAP']

print('='*80)
print('EXTENDED EVALUATION RESULTS (LightGBM)')
print('='*80)
for ds, df in results.items():
    print(f'\n--- {ds.upper()} ---')
    lgbm = df[df['model'] == 'lightgbm']
    for m in methods:
        s = lgbm[lgbm['method'] == m]
        if len(s) == 0: continue
        j = s['mean_jaccard'].dropna().mean()
        js = f'{j:.3f}' if not np.isnan(j) else ' ---'
        print(f'  {m:20s} n={s["n_features"].mean():5.1f} J={js:6s} '
              f'MCC={s["test_mcc"].mean():.3f}+/-{s["test_mcc"].std():.3f} '
              f'F1={s["test_f1"].mean():.3f} AUC={s["test_auc_roc"].mean():.3f}')

print('\n' + '='*80)
print('EXTENDED EVALUATION RESULTS (SVM)')
print('='*80)
for ds, df in results.items():
    print(f'\n--- {ds.upper()} ---')
    svm = df[df['model'] == 'svm']
    for m in methods:
        s = svm[svm['method'] == m]
        if len(s) == 0: continue
        j = s['mean_jaccard'].dropna().mean()
        js = f'{j:.3f}' if not np.isnan(j) else ' ---'
        print(f'  {m:20s} n={s["n_features"].mean():5.1f} J={js:6s} '
              f'MCC={s["test_mcc"].mean():.3f}+/-{s["test_mcc"].std():.3f} '
              f'F1={s["test_f1"].mean():.3f} AUC={s["test_auc_roc"].mean():.3f}')

print('\n' + '='*80)
print('WILCOXON: AEFS vs BorutaSHAP')
print('='*80)
for ds, df in results.items():
    for model in ['lightgbm', 'random_forest', 'svm']:
        aefs = df[(df['method']=='AEFS')&(df['model']==model)]['test_mcc'].values
        bor = df[(df['method']=='BorutaSHAP')&(df['model']==model)]['test_mcc'].values
        if len(aefs)==len(bor) and len(aefs)>2:
            try:
                stat, pval = stats.wilcoxon(aefs, bor, alternative='two-sided')
                sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else 'ns'
                delta = (sum(1 for a in aefs for b in bor if a>b)-sum(1 for a in aefs for b in bor if a<b))/(len(aefs)*len(bor))
                eff = 'negl.' if abs(delta)<0.147 else 'small' if abs(delta)<0.33 else 'medium' if abs(delta)<0.474 else 'large'
                print(f'  {ds:8s} {model:12s}: p={pval:.4f} {sig} delta={delta:+.3f} ({eff})')
            except: pass

print('\n' + '='*80)
print('FEATURE REDUCTION: AEFS vs BorutaSHAP')
print('='*80)
for ds, df in results.items():
    lgbm = df[df['model']=='lightgbm']
    aefs_n = lgbm[lgbm['method']=='AEFS']['n_features'].mean()
    bor_n = lgbm[lgbm['method']=='BorutaSHAP']['n_features'].mean()
    all_n = lgbm[lgbm['method']=='AllFeatures']['n_features'].mean()
    print(f'  {ds:8s}: All={all_n:.0f}  AEFS={aefs_n:.1f} ({100*(1-aefs_n/all_n):.0f}% reduction)  BorutaSHAP={bor_n:.0f} ({100*(1-bor_n/all_n):.0f}% reduction)')

print('\n' + '='*80)
print('AEFS vs SVM: IS AEFS ROBUST ACROSS CLASSIFIERS?')
print('='*80)
for ds, df in results.items():
    aefs_lgbm = df[(df['method']=='AEFS')&(df['model']=='lightgbm')]['test_mcc'].mean()
    aefs_svm = df[(df['method']=='AEFS')&(df['model']=='svm')]['test_mcc'].mean()
    aefs_rf = df[(df['method']=='AEFS')&(df['model']=='random_forest')]['test_mcc'].mean()
    print(f'  {ds:8s}: LGBM={aefs_lgbm:.3f}  RF={aefs_rf:.3f}  SVM={aefs_svm:.3f}')
