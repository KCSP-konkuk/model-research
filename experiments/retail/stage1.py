"""1단계 — 검증 2019~2021: 소매 자기이력 / +달력 / +가락 / 2변수 선형, h=1·2·3

  python stage1.py 붉은고추
"""
import sys
from sklearn.linear_model import LinearRegression
from common import *
from features import *

item = sys.argv[1]
df = build_frame(item)
print(f'{item}: {df.soon.iloc[0]} ~ {df.soon.iloc[-1]}  {len(df)}순')
G = {'retail': f_retail(df), 'cal': f_calendar(df), 'garak': f_garak(df)}


def lin(df, h, years):
    """'직전 순 가락 변화 + 직전 순 소매 변화' 선형 (비율 목표)"""
    X = pd.DataFrame({'g': df.g.shift(1) / df.g.shift(2), 'r': df.y.shift(1) / df.y.shift(2)})
    anchor, act = df.y.shift(1), df.y.shift(-(h - 1))
    ty = df.year.shift(-(h - 1))
    ok = X.notna().all(axis=1) & act.notna() & anchor.notna()
    out = []
    for Y in years:
        tr, te = ok & (ty < Y), ok & (ty == Y)
        m = LinearRegression().fit(X[tr], (act / anchor)[tr])
        out.append(pd.DataFrame({'pred': m.predict(X[te]) * anchor[te], 'act': act[te], 'nv': anchor[te],
                                 'year': Y, 'soon': df.soon[te], 'origin': df.index[te]}))
    return pd.concat(out, ignore_index=True)


for h in HORIZONS:
    print(f'\n--- h={h} (검증 {VAL[0]}~{VAL[-1]}) ---')
    show('선형 2변수(가락Δ+소매Δ)', lin(df, h, VAL), df)
    for name, groups in [('소매', ['retail']), ('소매+달력', ['retail', 'cal']),
                         ('소매+달력+가락', ['retail', 'cal', 'garak'])]:
        X = pd.concat([G[g] for g in groups], axis=1)
        show(name, run(df, X, h, VAL), df)
