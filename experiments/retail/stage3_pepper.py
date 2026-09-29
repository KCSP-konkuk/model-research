"""붉은고추 3단계 — 도매 예측 피쳐. 기준은 2단계 판정(±0.012) 반영: h=1 소매+가락, h=2 소매+달력+가락, h=3 소매+가락+교차+반입량"""
from common import *
from features import *

df = build_frame('붉은고추')
B = {1: [f_retail(df), f_garak(df)],
     2: [f_retail(df), f_calendar(df), f_garak(df)],
     3: [f_retail(df), f_garak(df), f_cross(df), f_supply(df, '붉은고추')]}
W = f_wholesale_pred(df)
for h in HORIZONS:
    X0 = pd.concat(B[h], axis=1)
    v0 = [mase(run(df, X0, h, VAL, seeds=4, seed0=100 * i)) for i in range(3)]
    v1 = [mase(run(df, pd.concat([X0, W], axis=1), h, VAL, seeds=4, seed0=100 * i)) for i in range(3)]
    print(f'h={h}  기준 {np.mean(v0):.3f}  +도매예측 {np.mean(v1):.3f} ({np.mean(v1) - np.mean(v0):+.3f})', flush=True)
