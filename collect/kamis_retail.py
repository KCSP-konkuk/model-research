"""KAMIS Open API 17번 소매(periodRetailProductList) 서울·상품 — 품목 × 연도 원본 JSON 저장

키는 레포에 없다. 서버 /opt/agri-forecast/application-secret.properties 의 kamis.cert-key·cert-id 두 줄을
로컬 파일로 복사해 KAMIS_PROPS 로 넘긴다. 이미 받은 연도 파일은 건너뛰어 이어받는다.
1회 최대 1년, 가끔 무응답 → 타임아웃 60초 + 3회 재시도. 2014~2026 품목 3개 = 39회 (2026-09-29 전부 1회에 성공)
애호박·시금치는 2026-09-30 추가(소매·도매 각 26회, 전부 1회에 성공). 품목만 고르려면 인자로: python kamis_retail.py raw/ 애호박 시금치

  KAMIS_PROPS=kamis.properties python kamis_retail.py raw/
  도매(16번 periodWholesaleProductList, 서울 = 가락도매 중도매인 판매가): KAMIS_ACTION=periodWholesaleProductList
  이어서 ../experiments/retail/parse_kamis.py 로 CSV
"""
import json, os, sys, time, datetime as dt
import requests

OUT = sys.argv[1] if len(sys.argv) > 1 else 'raw'
os.makedirs(OUT, exist_ok=True)
PROPS = os.environ['KAMIS_PROPS']
conf = dict(l.strip().split('=', 1) for l in open(PROPS) if '=' in l)
KEY, ID = conf['kamis.cert-key'], conf['kamis.cert-id']
ITEMS = {'양파': ('200', '245', '00'), '붉은고추': ('200', '243', '00'), '양배추': ('200', '212', '00'),
         '애호박': ('200', '224', '01'), '시금치': ('200', '213', '00')}
ONLY = sys.argv[2:] or list(ITEMS)
URL = 'https://www.kamis.or.kr/service/price/xml.do'
ACTION = os.environ.get('KAMIS_ACTION', 'periodRetailProductList')
GAP, TIMEOUT, ATTEMPTS = 1.5, 60, 3
today = dt.date.today()

last = 0.0
for name in ONLY:
    cat, item, kind = ITEMS[name]
    for y in range(2014, today.year + 1):
        path = os.path.join(OUT, f'{name}_{y}.json')
        if os.path.exists(path):
            continue
        end = min(dt.date(y, 12, 31), today)
        params = dict(action=ACTION, p_startday=f'{y}-01-01', p_endday=end.isoformat(),
                      p_itemcategorycode=cat, p_itemcode=item, p_kindcode=kind, p_productrankcode='04',
                      p_countrycode='1101', p_convert_kg_yn='N', p_cert_key=KEY, p_cert_id=ID, p_returntype='json')
        for attempt in range(1, ATTEMPTS + 1):
            time.sleep(max(0.0, GAP - (time.time() - last)))
            last = time.time()
            try:
                r = requests.get(URL, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                body = r.json()
                open(path, 'w', encoding='utf-8').write(json.dumps(body, ensure_ascii=False))
                n = len(body.get('data', {}).get('item', [])) if isinstance(body.get('data'), dict) else 0
                print(f'{name} {y}: {n} rows', flush=True)
                break
            except Exception as e:
                print(f'{name} {y}: fail {attempt}/{ATTEMPTS} {type(e).__name__}', flush=True)
                time.sleep(5)
print('finished', flush=True)
