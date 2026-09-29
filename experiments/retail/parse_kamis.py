"""KAMIS 소매 원본(raw/*.json) → data/retail/retail_seoul_2014.csv (item,date,market,price). market 은 판매처명 또는 평균·평년"""
import sys
RAW = sys.argv[1] if len(sys.argv) > 1 else 'raw'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'retail_seoul_2014.csv'
import json, glob, os, csv, datetime as dt, collections
out = []
for p in sorted(glob.glob(RAW + '/*.json')):
    name, y = os.path.basename(p)[:-5].rsplit('_', 1)
    y = int(y)
    data = json.load(open(p)).get('data')
    if not isinstance(data, dict): continue
    items = data.get('item', [])
    if isinstance(items, dict): items = [items]
    for it in items:
        md = (it.get('regday') or '').split('/')
        if it.get('yyyy') != str(y) or len(md) != 2: continue   # 요청 연도 밖 날짜는 그 해 파일에서 온 것만
        price = (it.get('price') or '').replace(',', '').strip()
        if not price or price == '-': continue
        county = it.get('countyname') or ''
        market = county if county in ('평균', '평년') else (it.get('marketname') or '')
        out.append((name, dt.date(y, int(md[0]), int(md[1])).isoformat(), market, int(price)))
out = sorted(set(out))
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['item', 'date', 'market', 'price']); w.writerows(out)
print(len(out), 'rows')
span = collections.defaultdict(list)
for n, d, m, p in out: span[(n, m)].append(d)
for (n, m), ds in sorted(span.items()):
    if m in ('경동', '복조리', '평균', '영등포'):
        print(f'{n:5s} {m:4s} {min(ds)} ~ {max(ds)}  {len(ds)}일')
# 중복 검사: 같은 품목·날짜·판매처에 값이 둘 이상
dup = collections.Counter((n, d, m) for n, d, m, p in out)
print('dup keys', sum(1 for v in dup.values() if v > 1))
