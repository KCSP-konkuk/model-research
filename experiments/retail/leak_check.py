"""피쳐 누수 검사 — 순 t 행의 피쳐가 순 t 이후 정보를 쓰지 않는지 (2026-09-29, pt_expected 누수를 찾은 뒤 추가)

1) 표 기반 함수: 순 t 와 그 뒤의 가격·집계를 NaN 으로 지우고 다시 계산해 순 t 행이 같은지
2) 외부 파일 함수: 원본 파일을 순 t 시작일 전날까지로 잘라 다시 계산해 순 t 행이 같은지
   (네이버 검색량은 파일 전체 최댓값=100 정규화라 이 방법으로 못 잡는다 — 수준 피쳐엔 미래 최댓값이 섞인다. 소매 모델은 안 씀)
새 피쳐 함수를 만들면 여기에 넣고 돌린다:  python leak_check.py
"""
import warnings
import pandas as pd
import numpy as np
import common as C
import features as F

warnings.filterwarnings('ignore')
POS = (100, 250, 400, -1)


def same(a, b):
    return [c for c in a.index if not (np.isclose(a[c], b[c]) or (np.isnan(a[c]) and np.isnan(b[c])))]


def check_frame(item, fns):
    df = C.build_frame(item)
    base = {n: fn(df).astype(float) for n, fn in fns.items()}
    bad = {}
    for i in POS:
        k = df.index[i]
        m = df.copy()
        num = [c for c in m.columns if c not in ('soon', 'year', 'month', 'pidx')]
        m.loc[m.index >= k, num] = np.nan
        for n, fn in fns.items():
            d = same(base[n].loc[k], fn(m).astype(float).loc[k])
            if d:
                bad.setdefault(n, set()).update(d)
    return bad


def check_file(item, fns):
    df = C.build_frame(item)
    real = pd.read_csv
    bad = {}
    for n, (fn, col) in fns.items():
        full = fn(df).astype(float)
        for i in POS:
            k = df.index[i]
            c = df.soon[k]
            t0 = pd.Timestamp(int(c[:4]), int(c[4:6]), {'상': 1, '중': 11, '하': 21}[c[6]])

            def cut(path, *a, **kw):
                d = real(path, *a, **kw)
                if col not in d.columns:
                    return d
                return d[pd.to_datetime(d[col].astype(str)) < t0]
            pd.read_csv = cut
            try:
                part = fn(df).astype(float).loc[k]
            finally:
                pd.read_csv = real
            d = same(full.loc[k], part)
            if d:
                bad.setdefault(n, set()).update(d)
    return bad


if __name__ == '__main__':
    frame_fns = {'f_retail': F.f_retail, 'f_garak': F.f_garak, 'f_retail_last': F.f_retail_last,
                 'f_calendar': F.f_calendar, 'f_garak_recent': F.f_garak_recent, 'f_market_split': F.f_market_split,
                 'f_passthrough': F.f_passthrough, 'f_cabbage_season': F.f_cabbage_season,
                 'f_holiday': lambda d: F.f_holiday(d, 1), 'f_onion_season': F.f_onion_season}
    ok = True
    for item in ('붉은고추', '양배추', '양파'):
        file_fns = {'f_bigmart': (lambda d, it=item: F.f_bigmart(d, it), 'date'),
                    'f_kamis_wholesale': (lambda d, it=item: F.f_kamis_wholesale(d, it), 'date')}
        if item == '양배추':
            file_fns['f_weather'] = (lambda d: F.f_weather(d, C.DATA + 'retail/cabbage_kma_daily.csv',
                                                           {184: '제주', 189: '서귀포', 216: '태백'}), 'tm')
        bad = {**check_frame(item, frame_fns), **check_file(item, file_fns)}
        print(item, '누수:', {k: sorted(v) for k, v in bad.items()} or '없음')
        ok &= not bad
    raise SystemExit(0 if ok else 1)
