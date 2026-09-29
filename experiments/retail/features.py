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


def f_wholesale_pred(df):
    """홍고추 도매 모델이 순 t 첫날 낸 이번 순 가락 예측(확장 윈도우 재현, wholesale_pred.py).
    가락이 어디로 갈지 → 소매가 따라갈 방향"""
    w = pd.read_csv(DATA + 'retail/wholesale_pred_홍고추.csv').set_index('DATE').pred
    p = df.soon.map(w)
    f = pd.DataFrame(index=df.index)
    f['wp_chg'] = p / df.g.shift(1)                 # 예측된 이번 순 가락 변화
    f['wp_vs_retail'] = p / df.y.shift(1)           # 예측 가락 대비 지금 소매 (마진이 어디로 가야 하나)
    f['wp_margin_dev'] = (df.y.shift(1) / p) / (df.y / df.g).shift(1).rolling(36, min_periods=18).mean()
    return f


def f_retail_last(df):
    """직전 순 막판 소매 — 순 안에서 이미 움직였으면 다음 순은 그 수준에서 시작한다"""
    f = pd.DataFrame(index=df.index)
    f['r_last_vs_mean'] = (df.r_last / df.y).shift(1)
    f['r_last3_vs_mean'] = (df.r_last3 / df.y).shift(1)
    return f


def f_market_split(df):
    """경동·복조리 따로: 각자 막판 값 ÷ 순 평균, 한쪽만 먼저 움직였는지"""
    f = pd.DataFrame(index=df.index)
    for m in ('경동', '복조리'):
        f[f'{m}_last_vs_y'] = (df[f'{m}_last'] / df.y).shift(1)
        f[f'{m}_chg1'] = (df[m] / df[m].shift(1)).shift(1)
    f['mkt_lead'] = f['경동_chg1'] / f['복조리_chg1']
    return f


def f_garak_recent(df):
    """예측 전날까지 최근 5·10 거래일 가락, 순 안 추세(막판 3일 ÷ 첫 3일)"""
    f = pd.DataFrame(index=df.index)
    f['g_tail5_vs_mean'] = (df.g_tail5 / df.g).shift(1)
    f['g_tail10_vs_mean'] = (df.g_tail10 / df.g).shift(1)
    f['g_tail5_vs_y'] = (df.g_tail5 / df.y).shift(1)        # 최신 가락 대비 소매 (마진)
    f['g_intra_trend'] = (df.g_last3 / df.g_first3).shift(1)
    return f


def f_passthrough(df, years_back=3):
    """계절별 전이율: 대상 순과 같은 달의 과거(직전 years_back 년) 순에서
    소매 변화 = β × 가락 변화(1순 시차) 의 β (원점 회귀). 모델이 가락 신호를 계절마다 얼마나 믿을지.
    쓰는 쌍은 전부 대상 순의 전년 이전이라 누수 없음"""
    dr = np.log(df.y / df.y.shift(1))            # 순 s 의 소매 변화
    dg = np.log(df.g.shift(1) / df.g.shift(2))   # 그 직전 순 가락 변화 (예측 때 아는 값과 같은 시차)
    beta, fit = [], []
    for i in df.index:
        y, m = df.year[i], df.month[i]
        pool = df.index[(df.month == m) & (df.year < y) & (df.year >= y - years_back)]
        a, b = dr.reindex(pool), dg.reindex(pool)
        ok = a.notna() & b.notna()
        if ok.sum() >= 4 and (b[ok] ** 2).sum() > 0:
            bt = (a[ok] * b[ok]).sum() / (b[ok] ** 2).sum()
            beta.append(bt)
            fit.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan)
        else:
            beta.append(np.nan); fit.append(np.nan)
    f = pd.DataFrame(index=df.index)
    f['pt_beta'] = beta
    f['pt_corr'] = fit
    f['pt_expected'] = np.exp(np.array(beta) * dg.shift(-1).values)   # 이번 순 기대 소매 비율 = exp(β × 직전 순 가락 변화)
    return f


# ---------- 양배추 ----------
# 산지 전환: 1~4월 제주·남해안 월동 → 5~6월 봄 평지 → 7~9월 강원 고랭지 → 10~12월 가을 평지 (농사로·제주 원예작물지도)
CABBAGE_REGION = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3}


def f_cabbage_season(df):
    """대상 순의 산지 구분 + 전환 직전·직후 순인지 (달력이라 미리 안다)"""
    f = pd.DataFrame(index=df.index)
    reg = df.month.map(CABBAGE_REGION)
    f['cb_region'] = reg
    f['cb_switch'] = (reg != reg.shift(1)).astype(int)            # 대상 순이 새 산지 첫 순
    f['cb_pre_switch'] = (reg.shift(-1) != reg).astype(int)       # 다음 순부터 산지가 바뀜
    return f


def f_weather(df, path, stations):
    """ASOS 일별 → 순 집계(평균기온·최저·강수합·일조), 직전 순 값과 같은 순 과거 평균 대비 편차.
    stations: {지점번호: 이름}. 대상 순 이전 순만 쓴다(shift 1)"""
    w = pd.read_csv(path)
    w['date'] = pd.to_datetime(w.tm.astype(str))
    w['soon'] = soon_code(w.date)
    f = pd.DataFrame(index=df.index)
    for stn, nm in stations.items():
        g = w[w.stn == stn].groupby('soon').agg(ta=('TA', 'mean'), tmin=('TMIN', 'min'),
                                                 rn=('RN', lambda x: x.fillna(0).sum()), ss=('SS', 'mean'))
        for c in g.columns:
            v = df.soon.map(g[c])
            v1 = v.shift(1)
            # 같은 순(36순 전들)의 과거 5년 평균 대비
            clim = pd.concat([v.shift(1 + 36 * j) for j in range(1, 6)], axis=1).mean(axis=1)
            f[f'w_{nm}_{c}'] = v1
            f[f'w_{nm}_{c}_dev'] = v1 - clim
            if c == 'rn':
                f[f'w_{nm}_rn3'] = v.shift(1).rolling(3).sum()
    return f


# ---------- KAMIS 안의 다른 가격 (2026-09-29) ----------
_TRAD = {'경동', '복조리', '영등포', '평균', '평년'}


def f_bigmart(df, item):
    """대형유통(A~L-유통 등) 소매: 판매처 구성이 자주 바뀌어 평균 대신 '같은 판매처의 순 평균 변화율' 중앙값.
    + 서울 평균 변화, 대형유통이 전통시장보다 먼저 움직였는지"""
    from common import RETAIL_CSV
    d = pd.read_csv(RETAIL_CSV, parse_dates=['date'])
    d = d[d.item == item].copy()
    d['soon'] = soon_code(d.date)
    big = d[~d.market.isin(_TRAD)].groupby(['soon', 'market']).price.mean().unstack()
    big = big.reindex(df.soon)
    big.index = df.index
    chg = big / big.shift(1)                              # 판매처별 순 변화 (둘 다 있을 때만)
    med = chg.median(axis=1, skipna=True)
    cnt = chg.notna().sum(axis=1)
    seoul = d[d.market == '평균'].groupby('soon').price.mean()
    s = df.soon.map(seoul)
    f = pd.DataFrame(index=df.index)
    f['bm_chg1'] = med.where(cnt >= 2).shift(1)
    f['bm_chg2'] = (med.where(cnt >= 2).shift(1) * med.where(cnt >= 2).shift(2))
    f['bm_n'] = cnt.shift(1)
    f['seoul_chg1'] = (s / s.shift(1)).shift(1)
    f['bm_lead'] = f['bm_chg1'] / (df.y / df.y.shift(1)).shift(1)
    f['trad_vs_seoul'] = (df.y / s).shift(1)
    return f


def f_kamis_wholesale(df, item):
    """KAMIS 도매(16번, 서울 가락도매 = 중도매인 판매가): 순 평균 변화, 막판, 소매/도매 마진, 도매 평년 대비"""
    w = pd.read_csv(DATA + 'retail/wholesale_seoul_2014.csv', parse_dates=['date'])
    w = w[w.item == item]
    w['soon'] = soon_code(w.date)
    gd = w[w.market == '가락도매'].sort_values('date')
    m = gd.groupby('soon').price.mean()
    last = gd.groupby('soon').price.last()
    ny = w[w.market == '평년'].groupby('soon').price.mean()
    v, vl, vn = df.soon.map(m), df.soon.map(last), df.soon.map(ny)
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3):
        f[f'kw_chg{k}'] = (v.shift(1) / v.shift(1 + k))
    f['kw_last_vs_mean'] = (vl / v).shift(1)
    f['kw_margin'] = (df.y / v).shift(1)
    f['kw_margin_dev36'] = f.kw_margin / f.kw_margin.rolling(36, min_periods=18).mean()
    f['kw_vs_ny'] = (v / vn).shift(1)
    f['kw_vs_garak'] = (v / df.g).shift(1)                 # 중도매인가 ÷ 경락가
    return f
