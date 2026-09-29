"""순 t 첫날 예측용 피처. 모든 값은 t-1 순까지의 정보로만 만든다 (shift(1) 이 기준)"""
import numpy as np
import pandas as pd


def f_retail(df, lags=(1, 2, 3, 6)):
    """소매 자기이력: 변화율, 평년·전년 대비, 판매처 간 차이, 경직성"""
    y1 = df.y.shift(1)
    f = pd.DataFrame(index=df.index)
    for k in lags:
        f[f'r_chg{k}'] = y1 / df.y.shift(1 + k)
    f['r_yoy'] = y1 / df.y.shift(1 + 36)
    # 평년: 직전 5년 같은 순 평균 (자기 해 제외 — 전부 과거값)
    ny = pd.concat([df.y.shift(1 + 36 * j) for j in range(1, 6)], axis=1).mean(axis=1, skipna=True)
    f['r_vs_ny'] = y1 / ny
    # 대상 순의 평년 ÷ 지금 가격: 계절이 어디로 가는지 (h=1 기준 대상 = t, 1년 전 t 부터 5년)
    ny_t = pd.concat([df.y.shift(36 * j) for j in range(1, 6)], axis=1).mean(axis=1, skipna=True)
    f['r_ny_next'] = ny_t / y1
    f['r_mkt_gap'] = (df.경동 / df.복조리).shift(1)
    f['r_same'] = df.r_same.shift(1)
    moved = ((df.y / df.y.shift(1) - 1).abs() > 0.02).astype(float).shift(1)
    f['r_still3'] = 3 - moved.rolling(3).sum()
    return f


def f_garak(df, lags=(1, 2, 3)):
    """가락 경매가: 순 평균 변화, 막판 가격, 오름·내림 분리, 소매/도매 마진 편차"""
    g1 = df.g.shift(1)
    f = pd.DataFrame(index=df.index)
    for k in lags:
        c = g1 / df.g.shift(1 + k) - 1
        f[f'g_chg{k}'] = c
        if k == 1:
            f['g_up1'] = c.clip(lower=0)
            f['g_dn1'] = c.clip(upper=0)
    f['g_last_vs_mean'] = (df.g_last / df.g).shift(1)          # 홍고추 d_last1_vs_p1
    f['g_last3_vs_mean'] = (df.g_last3 / df.g).shift(1)
    f['g_yoy'] = g1 / df.g.shift(1 + 36)
    margin = df.y / df.g
    m1 = margin.shift(1)
    f['margin'] = m1
    f['margin_dev36'] = m1 / m1.rolling(36, min_periods=18).mean()   # 1년 평균 대비
    f['margin_dev6'] = m1 / m1.rolling(6, min_periods=3).mean()
    # 도매는 움직였는데 소매는 아직: 가락 누적 변화 - 소매 누적 변화 (3순)
    f['gap3'] = (g1 / df.g.shift(4)) / (df.y.shift(1) / df.y.shift(4))
    return f


def f_calendar(df):
    f = pd.DataFrame(index=df.index)
    f['month'] = df.month
    f['pidx'] = df.pidx
    f['soon36'] = (df.month - 1) * 3 + df.pidx
    return f


# ---------- 품목별 추가 후보 ----------
import os
from common import DATA, soon_code

SEOL = {y: m for y, m in zip(range(2014, 2027), ['0131', '0219', '0208', '0128', '0216', '0205', '0125',
                                                 '0212', '0201', '0122', '0210', '0129', '0217'])}
CHU = {y: m for y, m in zip(range(2014, 2027), ['0908', '0927', '0915', '1004', '0924', '0913', '1001',
                                                '0921', '0910', '0929', '0917', '1006', '0925'])}


def _soon_series(path, col):
    """농넷 순별 CSV(DATE='201407상순') → 우리 순 코드 '201407상' 기준 Series"""
    s = pd.read_csv(path)
    return pd.Series(pd.to_numeric(s[col], errors='coerce').values, index=s.DATE.str[:7])


def f_cross(df, items=('풋고추', '청피망'), lags=(1, 2, 3)):
    """고추류 교차 가격(가락 순별). 홍고추 도매에서 가장 큰 지렛대였다"""
    f = pd.DataFrame(index=df.index)
    for it in items:
        v = df.soon.map(_soon_series(DATA + f'cross_long_{it}.csv', 'val'))
        v1 = v.shift(1)
        f[f'x_{it}_ratio'] = df.g.shift(1) / v1
        for k in lags:
            f[f'x_{it}_chg{k}'] = v1 / v.shift(1 + k)
    return f


def f_supply(df, item):
    """가락 반입량(순별 합계): 변화, 전년·평년 대비"""
    path = DATA + ('area_long_홍고추.csv' if item == '붉은고추' else f'retail/area_{item}.csv')
    if not os.path.exists(path):
        return pd.DataFrame(index=df.index)
    s = df.soon.map(_soon_series(path, 'sup'))
    f = pd.DataFrame(index=df.index)
    s1 = s.shift(1)
    f['sup_chg1'] = s1 / s.shift(2)
    f['sup_yoy'] = s1 / s.shift(37)
    f['sup_vs_ma6'] = s1 / s.shift(1).rolling(6).mean()
    return f


def f_trend(df, kw):
    """네이버 검색량(2016~). 2014~2015 는 비어 있다(트리는 결측을 그대로 받는다)"""
    t = pd.read_csv(DATA + 'pepper_trend.csv')
    t = t[t.kw == kw]
    s = pd.Series(t.ratio.values, index=pd.to_datetime(t.period))
    m = s.groupby(soon_code(s.index)).mean()
    v = df.soon.map(m)
    f = pd.DataFrame(index=df.index)
    f['sr_ma3'] = v.shift(1).rolling(3).mean()
    f['sr_mom'] = v.shift(1) / v.shift(2)
    f['sr_yoy'] = v.shift(1) / v.shift(37)
    return f


def f_holiday(df, horizon=1):
    """대상 순(t+h-1)에 설·추석이 걸리는지, 명절 직전 순인지 — 달력이라 미리 안다"""
    def hit(tbl, k):
        out = []
        for i in df.index + k:
            y, m, p = i // 36, (i % 36) // 3 + 1, i % 3
            d = tbl.get(int(y), '')
            out.append(int(bool(d) and int(d[:2]) == m and (0 if int(d[2:]) <= 10 else 1 if int(d[2:]) <= 20 else 2) == p))
        return out
    k = horizon - 1
    f = pd.DataFrame(index=df.index)
    f['h_seol'] = hit(SEOL, k)
    f['h_chu'] = hit(CHU, k)
    f['h_pre_seol'] = hit(SEOL, k + 1)
    f['h_pre_chu'] = hit(CHU, k + 1)
    f['h_kimjang'] = np.isin(((df.index + k) % 36) // 3 + 1, [11, 12]).astype(int)
    return f
