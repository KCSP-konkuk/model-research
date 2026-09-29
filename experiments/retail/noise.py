"""시드 노이즈 폭 — 같은 구성을 시드 묶음(4개)만 바꿔 5번. 검증 2019~2021

  python noise.py 붉은고추
"""
import sys
from common import *
from features import *

item = sys.argv[1]
df = build_frame(item)
X = pd.concat([f_retail(df), f_calendar(df), f_garak(df)], axis=1)
for h in HORIZONS:
    v = [mase(run(df, X, h, VAL, seeds=4, seed0=100 * i)) for i in range(5)]
    print(f'h={h}  {" ".join(f"{x:.3f}" for x in v)}  평균 {np.mean(v):.3f}  σ {np.std(v, ddof=1):.4f}', flush=True)
