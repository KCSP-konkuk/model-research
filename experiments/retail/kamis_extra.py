"""KAMIS 안의 다른 가격(대형유통 소매·도매 중도매인가)을 확정 구성에 더하면? 검증 2017~2021, 기본 파라미터, 시드 묶음 3개
  python kamis_extra.py 붉은고추 [h]"""
import sys
from common import *
from features import *

item = sys.argv[1]
hs = [int(sys.argv[2])] if len(sys.argv) > 2 else [1]
VAL5 = (2017, 2018, 2019, 2020, 2021)
df = build_frame(item)
X0 = pd.concat([f_retail(df), f_garak(df), f_retail_last(df)], axis=1)
C = {'bigmart': f_bigmart(df, item), 'kamis_wh': f_kamis_wholesale(df, item)}
C['둘 다'] = pd.concat(C.values(), axis=1)


def m3(X, h):
    return np.mean([mase(run(df, X, h, VAL5, seeds=4, seed0=100 * i)) for i in range(3)])


for h in hs:
    b = m3(X0, h)
    print(f'{item} h={h}  확정 구성(소매+가락+소매막판) {b:.3f}', flush=True)
    for nm, Xc in C.items():
        print(f'  + {nm:<9} 결측률 {Xc.isna().mean().mean():.2f}', end='  ', flush=True)
        m = m3(pd.concat([X0, Xc], axis=1), h)
        print(f'{m:.3f} ({m - b:+.3f}) → {"채택" if m <= b - 0.012 else "제외"}', flush=True)
