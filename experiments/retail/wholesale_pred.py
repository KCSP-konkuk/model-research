"""홍고추 도매 모델(METHOD 11절 'S 선택')의 순별 예측을 2014~2026 확장 윈도우로 재현 → 소매 모델 피쳐

연도 Y 의 예측 = 2001 ~ Y-1 로 학습한 모델이 Y 의 각 순 첫날 낸 값 (그 시점에 알 수 있던 예측과 같은 조건)
  python wholesale_pred.py  → ../../data/retail/wholesale_pred_홍고추.csv (DATE, pred, act)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
os.chdir(os.path.join(HERE, '..'))
os.environ.setdefault('SEEDS', '4')
from daily_anchor import *

b = json.load(open('best_split.json'))['S 선택']
G = {'price': f_price(df), 'cross': f_cross_l(df), 'trend': f_trend(df), 'dlast': DF[['d_last1_vs_p1']]}
X = pd.concat([G[g] for g in b['groups']], axis=1)
P1 = df.price.shift(1)
R = run(X, P1, P1, years=tuple(range(2014, 2027)), seeds=int(os.environ['SEEDS']), params=b['params'], k=b['K'])
out = pd.DataFrame({'DATE': df.DATE[R.index].str[:7].values, 'pred': R.pred.values, 'act': R.act.values})
out.to_csv(os.path.join(HERE, '..', '..', 'data', 'retail', 'wholesale_pred_홍고추.csv'), index=False)
for y, g in R.groupby('yr'):
    print(y, f'MASE {np.abs(g.pred - g.act).mean() / np.abs(g.nv - g.act).mean():.3f}', flush=True)
