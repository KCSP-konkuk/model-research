"""붉은고추 4단계 — 막판·최근 피쳐 (검증 2019~2021, 기본 파라미터, 시드 묶음 3개 평균)
기준: h1 소매+가락+도매예측 / h2 +달력 / h3 +교차+반입량 (3단계와 같음)"""
from common import *
from features import *

df = build_frame('붉은고추')
W = f_wholesale_pred(df)
B = {1: [f_retail(df), f_garak(df), W],
     2: [f_retail(df), f_calendar(df), f_garak(df), W],
     3: [f_retail(df), f_garak(df), f_cross(df), f_supply(df, '붉은고추'), W]}
C = {'retail_last': f_retail_last(df), 'garak_recent': f_garak_recent(df), 'market_split': f_market_split(df)}
C['retail_last+garak_recent'] = pd.concat([C['retail_last'], C['garak_recent']], axis=1)


def m3(X, h):
    return np.mean([mase(run(df, X, h, VAL, seeds=4, seed0=100 * i)) for i in range(3)])


for h in HORIZONS:
    X0 = pd.concat(B[h], axis=1)
    b = m3(X0, h)
    print(f'\n--- h={h}  기준 {b:.3f}', flush=True)
    for name, Xc in C.items():
        m = m3(pd.concat([X0, Xc], axis=1), h)
        print(f'  + {name:<26} {m:.3f} ({m - b:+.3f})', flush=True)
