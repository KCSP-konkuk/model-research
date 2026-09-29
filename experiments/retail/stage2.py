"""2단계 — 검증 2019~2021: 기준(소매+달력+가락)에 후보를 하나씩 더해 h 마다 MASE 변화

  python stage2.py 붉은고추
"""
import sys
from common import *
from features import *

item = sys.argv[1]
df = build_frame(item)
SEEDS = 4
base = {'retail': f_retail(df), 'cal': f_calendar(df), 'garak': f_garak(df)}
cands = {'supply': lambda h: f_supply(df, item), 'holiday': lambda h: f_holiday(df, h)}
if item == '붉은고추':
    cands['cross'] = lambda h: f_cross(df)
    cands['trend'] = lambda h: f_trend(df, '고춧가루')

for h in HORIZONS:
    X0 = pd.concat(base.values(), axis=1)
    b = mase(run(df, X0, h, VAL, seeds=SEEDS))
    print(f'\n--- h={h}  기준 {b:.3f}', flush=True)
    for name, fn in cands.items():
        Xc = fn(h)
        if Xc.shape[1] == 0:
            print(f'  + {name:<8} (데이터 없음)')
            continue
        R = run(df, pd.concat([X0, Xc], axis=1), h, VAL, seeds=SEEDS)
        m = mase(R)
        print(f'  + {name:<8} {m:.3f} ({m - b:+.3f})  [{" ".join(f"{v:.2f}" for v in by_year(R).values())}]', flush=True)
    # 빼기: 기준 그룹 각각
    for g in base:
        R = run(df, pd.concat([v for k, v in base.items() if k != g], axis=1), h, VAL, seeds=SEEDS)
        m = mase(R)
        print(f'  - {g:<8} {m:.3f} ({m - b:+.3f})', flush=True)
