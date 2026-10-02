"""서울 전통시장 소매가 순별 예측 — 공통 틀 (2026-09-29)

목표: 경동·복조리 일별 소매가(KAMIS 17번, 상품) → 값이 있는 곳만 평균 → 순 평균. 2014-07 상순~
예측 시점: 순 t 의 첫날. 쓸 수 있는 값 = 소매 t-1 순까지, 가락 경매가 t-1 순 마지막 거래일까지
예측 기간 h (direct, h 마다 따로 학습): h=1 → 순 t, h=2 → t+1, h=3 → t+2
  목표 = y[t+h-1] / y[t-1] (비율 — 트리가 새 가격대에서 무너지지 않게), 나이브 = y[t-1] 유지
분할: 학습 ~2018 / 검증 2019~2021 (선택·튜닝은 여기서만) / 시험 2022~2025 (마지막 1회) / 추가 2026
  연도 Y 평가 = '목표 순'이 Y 이전인 행으로 학습(확장 윈도우). h>1 에서 목표가 Y 에 걸친 행이 학습에 새지 않게
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data') + os.sep
RETAIL_CSV = DATA + 'retail/retail_seoul_2014.csv'
MARKETS = ('경동', '복조리')
START = '201407상'
VAL = (2019, 2020, 2021)
TEST = (2022, 2023, 2024, 2025)
EXTRA = (2026,)
HORIZONS = (1, 2, 3)
FLAT = 0.02   # 방향 3분류의 '비슷' 폭(±2%)

# 소매 품목 → 가락 일별 파일 (상 등급)
GARAK = {'붉은고추': DATA + 'daily_홍고추.csv',
         '양파': DATA + 'retail/garak_daily_양파.csv',
         '양배추': DATA + 'retail/garak_daily_양배추.csv'}
# 가락 경매가 대신 KAMIS 16번 도매(서울 가락도매 = 중도매인 판매가)를 쓰는 품목 (2026-09-30, RETAIL.md 12절)
# 기존 3품목에서 경매가 자리에 넣어도 검증 MASE 가 ±0.03 안이었다 → 농넷 백필 없이 품목을 늘린다
KAMIS_G = ('애호박', '시금치', '오이')
WHOLESALE_CSV = DATA + 'retail/wholesale_seoul_2014.csv'

PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
              objective='reg:absoluteerror')


# ---------- 순 ----------
def soon_code(d):
    """DatetimeIndex → '201407상' 배열"""
    d = pd.DatetimeIndex(d)
    p = np.where(d.day <= 10, '상', np.where(d.day <= 20, '중', '하'))
    return np.array([f'{y}{m:02d}{s}' for y, m, s in zip(d.year, d.month, p)])


def soon_index(codes):
    """'201407상' → 연속 정수 (연*36 + (월-1)*3 + 순)"""
    c = pd.Series(codes)
    return (c.str[:4].astype(int) * 36 + (c.str[4:6].astype(int) - 1) * 3
            + c.str[6].map({'상': 0, '중': 1, '하': 2})).values


# ---------- 데이터 ----------
def load_retail_daily(item):
    d = pd.read_csv(RETAIL_CSV, parse_dates=['date'])
    d = d[(d.item == item) & d.market.isin(MARKETS)]
    w = d.pivot_table(index='date', columns='market', values='price')
    return w.reindex(columns=list(MARKETS))


def load_garak_daily(item):
    if item in KAMIS_G:
        w = pd.read_csv(WHOLESALE_CSV, parse_dates=['date'])
        return w[(w.item == item) & (w.market == '가락도매')].set_index('date')['price'].astype(float).sort_index()
    g = pd.read_csv(GARAK[item], parse_dates=['date'])
    return g.set_index('date')['상'].dropna().sort_index()


def build_frame(item):
    """순 단위 표. y = 경동·복조리 평균(있는 곳만)의 순 평균. 가락 g = 거래일 상 가격 순 평균"""
    r = load_retail_daily(item)
    r['avg'] = r.mean(axis=1)
    r['soon'] = soon_code(r.index)
    rs = r.groupby('soon').agg(y=('avg', 'mean'), 경동=('경동', 'mean'), 복조리=('복조리', 'mean'),
                               r_days=('avg', 'count'))
    # 소매 경직성: 순 안에서 판매처 값이 전날과 같았던 비율
    same = (r[list(MARKETS)].diff() == 0).where(r[list(MARKETS)].notna() & r[list(MARKETS)].shift().notna())
    rs['r_same'] = same.mean(axis=1).groupby(r['soon']).mean()
    # 순 막판 소매: 마지막 조사일·마지막 3조사일 평균, 판매처별 마지막 값
    rs['r_last'] = r.groupby('soon')['avg'].last()
    rs['r_last3'] = r.groupby('soon')['avg'].apply(lambda s: s.dropna().tail(3).mean())
    rs['경동_last'] = r.groupby('soon')['경동'].last()
    rs['복조리_last'] = r.groupby('soon')['복조리'].last()

    g = load_garak_daily(item).to_frame('g')
    g['soon'] = soon_code(g.index)
    gs = g.groupby('soon').g.agg(g='mean', g_last='last', g_first='first', g_days='count')
    gs['g_last3'] = g.groupby('soon').g.apply(lambda s: s.tail(3).mean())
    gs['g_first3'] = g.groupby('soon').g.apply(lambda s: s.head(3).mean())
    # 순 경계를 넘는 '예측 전날까지 최근 n 거래일' (순 t 첫날 기준 = t-1 순 마지막 거래일까지)
    last_day = g.groupby('soon').apply(lambda x: x.index.max())
    gv = g.g
    for n in (5, 10):
        gs[f'g_tail{n}'] = [gv[gv.index <= d].tail(n).mean() for d in last_day.reindex(gs.index)]

    df = rs.join(gs, how='left')
    # 진행 중인 순(오늘이 속한 순)은 값이 덜 찼으니 뺀다
    today = soon_code(pd.DatetimeIndex([pd.Timestamp.today()]))[0]
    df = df[(df.index >= START) & (df.index < today)]
    k = soon_index(df.index)
    full = pd.Index(range(k.min(), k.max() + 1))
    df.index = k
    df = df.reindex(full)
    df['soon'] = [f'{i // 36}{(i % 36) // 3 + 1:02d}{"상중하"[i % 3]}' for i in df.index]
    df['year'] = df.index // 36
    df['month'] = (df.index % 36) // 3 + 1
    df['pidx'] = df.index % 3
    return df


# ---------- 모델·평가 ----------
def make_model(params=None, seed=0):
    p = dict(PARAMS)
    p.update(params or {})
    return xgb.XGBRegressor(**p, random_state=seed)


def run(df, X, h, years, k=None, seeds=4, params=None, seed0=0):
    """행 t = 순 t 첫날의 예측. X 의 행 t 는 t-1 순까지의 정보로만 만들어져 있어야 한다.
    반환: 평가 연도 행마다 pred·act·nv(나이브)·year·soon"""
    y = df.y
    anchor = y.shift(1)                   # y[t-1]
    act = y.shift(-(h - 1))               # y[t+h-1]
    tgt_year = df.year.shift(-(h - 1))
    ratio = act / anchor
    X = X.astype(float).replace([np.inf, -np.inf], np.nan)
    ok = ratio.notna() & np.isfinite(ratio) & anchor.notna()
    out = []
    for Y in years:
        tr = ok & (tgt_year < Y)
        te = ok & (tgt_year == Y)
        if te.sum() == 0:
            continue
        cols = list(X.columns)
        if k and k < len(cols):
            m0 = make_model(params, 0).fit(X.loc[tr, cols], ratio[tr])
            cols = list(pd.Series(m0.feature_importances_, index=cols).nlargest(k).index)
        ps = [make_model(params, s).fit(X.loc[tr, cols], ratio[tr]).predict(X.loc[te, cols])
              for s in range(seed0, seed0 + seeds)]
        out.append(pd.DataFrame({'pred': np.mean(ps, axis=0) * anchor[te].values, 'act': act[te].values,
                                 'nv': anchor[te].values, 'year': Y, 'soon': df.soon[te].values,
                                 'origin': df.index[te]}))
    return pd.concat(out, ignore_index=True)


def direction(new, base, flat=FLAT):
    r = new / base - 1
    return np.where(r > flat, 1, np.where(r < -flat, -1, 0))


def panel(R):
    """논문 관행 지표 묶음 + NMAE(aT 대회) + 3분류 방향. 나이브 = 직전 순 유지"""
    p, a, n = R.pred.values, R.act.values, R.nv.values
    e = p - a
    ae, ne = np.abs(e), np.abs(n - a)
    da_true = direction(a, n)
    da_pred = direction(p, n)
    return dict(MASE=ae.mean() / ne.mean(), RMSSE=np.sqrt((e ** 2).mean() / ((n - a) ** 2).mean()),
                MAE=ae.mean(), MAPE=np.mean(ae / a) * 100, MdAPE=np.median(ae / a) * 100,
                NMAE=ae.sum() / np.abs(a).sum() * 100, naive_MAPE=np.mean(ne / a) * 100,
                DA3=np.mean(da_pred == da_true) * 100,
                DA3_naive=np.mean(da_true == 0) * 100,    # 나이브는 항상 '비슷'
                moved=np.mean(da_true != 0) * 100,
                R2=1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2))


def by_year(R):
    return {y: np.abs(g.pred - g.act).mean() / np.abs(g.nv - g.act).mean() for y, g in R.groupby('year')}


def mase(R):
    return np.abs(R.pred - R.act).mean() / np.abs(R.nv - R.act).mean()


def dm(R, other=None, h=1, loss='abs'):
    """Diebold-Mariano (HLN 보정). other 가 없으면 나이브 대비"""
    a = R.act.values
    p2 = R.nv.values if other is None else other.pred.values
    if loss == 'abs':
        d = np.abs(R.pred.values - a) - np.abs(p2 - a)
    else:
        d = (R.pred.values - a) ** 2 - (p2 - a) ** 2
    n = len(d)
    s = d.mean() / np.sqrt(d.var(ddof=0) / n)
    hln = s * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return hln, 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))


def regime(R, df, q=0.7):
    """가락이 크게 움직인 순(직전 순 가락 변화율 절대값 상위 30%) vs 나머지의 MASE"""
    gchg = (df.g.shift(1) / df.g.shift(2) - 1).abs()
    v = gchg.reindex(R.origin).values
    cut = np.nanquantile(gchg, q)
    big = v >= cut
    return dict(big=mase(R[big]), calm=mase(R[~big]), big_n=int(big.sum()))


def show(name, R, df=None):
    m = panel(R)
    yr = ' '.join(f'{v:.2f}' for v in by_year(R).values())
    extra = ''
    if df is not None:
        rg = regime(R, df)
        extra = f"  큰변동 {rg['big']:.3f}/평온 {rg['calm']:.3f}"
    print(f"{name:<30} MASE {m['MASE']:.3f}  MAPE {m['MAPE']:5.2f}% (나이브 {m['naive_MAPE']:5.2f}%)  "
          f"NMAE {m['NMAE']:5.2f}%  DA3 {m['DA3']:.0f}% (나이브 {m['DA3_naive']:.0f}%)  [{yr}]{extra}", flush=True)
    return m
