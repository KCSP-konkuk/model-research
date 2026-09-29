"""붉은고추 — 검증 5년(2017~2021)으로 넓혀 피쳐 확인·튜닝 → 상위 5 파라미터 앙상블 → 시험 2022~2025 1회

배경: 검증 3년(2019~21)에서 고른 튜닝이 시험으로 옮겨지지 않았다(METHOD 소매 절). 표본을 늘리고 앙상블로 과적합을 줄인다
1) 피쳐 후보(h 마다): 기본 3종 + 각 후보 그룹 on/off, 기본 파라미터·시드 4 → 검증 MASE. ±0.012 넘을 때만 채택
2) 튜닝: 파라미터 40개 × K(전부·30·15), 시드 4 → 상위 5 조합 앙상블(예측 평균)
3) 시험 2022~2025 + 2026, 시드 12(조합마다), 나이브 대비 DM, 기본값 대비 DM
"""
import json, random
from common import *
from features import *

VAL5 = (2017, 2018, 2019, 2020, 2021)
TOL = 0.012
df = build_frame('붉은고추')
G = {'retail': f_retail(df), 'cal': f_calendar(df), 'garak': f_garak(df), 'wpred': f_wholesale_pred(df),
     'rlast': f_retail_last(df), 'cross': f_cross(df), 'supply': f_supply(df, '붉은고추'),
     'grecent': f_garak_recent(df), 'hol1': f_holiday(df, 1), 'hol2': f_holiday(df, 2), 'hol3': f_holiday(df, 3)}
CORE = ['retail', 'garak']


def X_of(gs):
    return pd.concat([G[g] for g in gs], axis=1)


def val(gs, h, params=None, k=None, seeds=4):
    return mase(run(df, X_of(gs), h, VAL5, k=k, seeds=seeds, params=params))


best = {}
for h in HORIZONS:
    sel = list(CORE)
    b = val(sel, h)
    print(f'\n=== h={h}  핵심(소매+가락) {b:.3f}', flush=True)
    for g in ['wpred', 'rlast', 'cal', 'cross', 'supply', 'grecent', f'hol{h}']:   # 앞에서 효과 컸던 순
        m = val(sel + [g], h)
        add = m <= b - TOL
        print(f'  + {g:<8} {m:.3f} ({m - b:+.3f}) → {"추가" if add else "제외"}', flush=True)
        if add:
            sel.append(g); b = m
    rng = random.Random(11)
    cands = [{}] + [dict(max_depth=rng.choice([2, 3, 4, 5]), n_estimators=rng.choice([300, 600, 1000]),
                         learning_rate=rng.choice([0.02, 0.03, 0.05, 0.08]), subsample=rng.choice([0.6, 0.8, 1.0]),
                         colsample_bytree=rng.choice([0.5, 0.7, 0.9]), min_child_weight=rng.choice([1, 3, 6, 12]),
                         reg_lambda=rng.choice([0.5, 1.0, 3.0])) for _ in range(39)]
    res = sorted((val(sel, h, p, k), k, i) for k in (None, 30, 15) for i, p in enumerate(cands)
                 if not (k and k >= X_of(sel).shape[1]))
    top = res[:5]
    best[h] = dict(groups=sel, top=[dict(K=k, params=cands[i], val=m) for m, k, i in top], val_default=b)
    print(f'  구성 {sel}  기본값 {b:.3f} → 상위5 {[round(t[0], 3) for t in top]}', flush=True)
json.dump(best, open('best_pepper5.json', 'w'), ensure_ascii=False, indent=1)


def ens(h, years, seeds=12):
    b = best[h]
    Rs = [run(df, X_of(b['groups']), h, years, k=t['K'], seeds=seeds, params=t['params']) for t in b['top']]
    R = Rs[0].copy()
    R['pred'] = np.mean([r.pred.values for r in Rs], axis=0)
    return R


print('\n=== 검증 5년 앙상블 확인 ===', flush=True)
for h in HORIZONS:
    show(f'h={h} 검증 앙상블', ens(h, VAL5, seeds=4), df)
print('\n=== 시험 2022~2025 · 추가 2026 (1회) ===', flush=True)
for h in HORIZONS:
    for years, label in ((TEST, '시험'), (EXTRA, '2026')):
        R = ens(h, years)
        R0 = run(df, X_of(best[h]['groups']), h, years, seeds=12)
        show(f'h={h} {label} 앙상블', R, df)
        show(f'h={h} {label} 기본값', R0, df)
        t, p = dm(R, h=h); t0, p0 = dm(R, R0, h=h)
        print(f'    DM 나이브 대비 t={t:.2f} p={p:.2g} / 기본값 대비 t={t0:.2f} p={p0:.2g}', flush=True)
        R.to_csv(f'pred_pepper5_h{h}_{label}.csv', index=False)
