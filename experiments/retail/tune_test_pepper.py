"""붉은고추 — 튜닝(검증 2019~2021) → 시험 2022~2025 1회 + 추가 2026

튜닝: 파라미터 후보 30개(고정 시드로 뽑음) × K(전부·30·15), 시드 4. h 마다 검증 MASE 최소를 고른다
시험: 고른 구성으로 시드 12. 나이브 대비 DM, 기준(파라미터 기본값) 대비 DM
"""
import json, random
from common import *
from features import *

df = build_frame('붉은고추')
W = f_wholesale_pred(df)
X = {1: pd.concat([f_retail(df), f_garak(df), W], axis=1),
     2: pd.concat([f_retail(df), f_calendar(df), f_garak(df), W], axis=1),
     3: pd.concat([f_retail(df), f_garak(df), f_cross(df), f_supply(df, '붉은고추'), W], axis=1)}
rng = random.Random(7)
CANDS = [{}] + [dict(max_depth=rng.choice([2, 3, 4, 5]), n_estimators=rng.choice([300, 600, 1000]),
                     learning_rate=rng.choice([0.02, 0.03, 0.05, 0.08]), subsample=rng.choice([0.6, 0.8, 1.0]),
                     colsample_bytree=rng.choice([0.5, 0.7, 0.9]), min_child_weight=rng.choice([1, 3, 6, 12]),
                     reg_lambda=rng.choice([0.5, 1.0, 3.0])) for _ in range(29)]
best = {}
for h in HORIZONS:
    res = []
    for k in (None, 30, 15):
        for i, p in enumerate(CANDS):
            m = mase(run(df, X[h], h, VAL, k=k, seeds=4, params=p))
            res.append((m, k, i))
    res.sort(key=lambda r: r[0])
    base = [r for r in res if r[1] is None and r[2] == 0][0][0]
    m, k, i = res[0]
    best[h] = dict(K=k, params=CANDS[i], val=m, val_default=base)
    print(f'h={h} 검증: 기본값 {base:.3f} → 최적 {m:.3f} (K={k}, 후보 {i})  상위5 {[f"{r[0]:.3f}" for r in res[:5]]}', flush=True)
json.dump(best, open('best_pepper.json', 'w'), ensure_ascii=False, indent=1)

print('\n=== 시험 2022~2025 (시드 12) ===', flush=True)
for h in HORIZONS:
    b = best[h]
    for years, label in ((TEST, '시험'), (EXTRA, '2026')):
        R = run(df, X[h], h, years, k=b['K'], seeds=12, params=b['params'])
        R0 = run(df, X[h], h, years, seeds=12)
        m = show(f'h={h} {label} 튜닝', R, df)
        show(f'h={h} {label} 기본값', R0, df)
        t, p = dm(R, h=h)
        t0, p0 = dm(R, R0, h=h)
        print(f'    DM 나이브 대비 t={t:.2f} p={p:.2g} / 기본값 대비 t={t0:.2f} p={p0:.2g}', flush=True)
        R.to_csv(f'pred_pepper_h{h}_{label}.csv', index=False)
