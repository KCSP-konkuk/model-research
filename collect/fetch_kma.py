"""ASOS 일별 수집 — 기간조회(kma_sfcdd3.php), 지점×연 1회씩

기본은 홍고추 산지 6지점 2001~ (54회). 다른 품목은 환경변수로:
  KMA_STNS=184:제주,189:서귀포,216:태백  KMA_START=2013  KMA_OUT=../data/retail/cabbage_kma_daily.csv
  KMA_PROPS=키 파일 (기본: 서버 설정 파일. weather.auth-key 한 줄이면 된다)
"""
import os, sys, time, requests, pandas as pd, numpy as np
from datetime import date, timedelta
URL='https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php'
STNS={192:'진주',156:'광주',212:'홍천',100:'대관령',211:'인제',136:'안동'}
if os.environ.get('KMA_STNS'):
    STNS={int(a):b for a,b in (x.split(':') for x in os.environ['KMA_STNS'].split(','))}
START=int(os.environ.get('KMA_START',2001))
END=int(os.environ.get('KMA_END',date.today().year))
OUT=os.environ.get('KMA_OUT','/tmp/pepper_kma_daily.csv')
IDX=dict(WS=2,WSMAX=5,TA=10,TMAX=11,TMIN=13,TS=16,TG=17,HM=18,HMMIN=19,CA=31,SS=32,SI=35,RN=38,RNDUR=40,EV=22)
MISS={-9.0,-99.0,-999.0,-9.9}
P={}
for line in open(os.environ.get('KMA_PROPS','/opt/agri-forecast/application-secret.properties')):
    if '=' in line and not line.strip().startswith('#'):
        k,_,v=line.strip().partition('='); P[k.strip()]=v.strip()
key=P['weather.auth-key']
rows=[]; today=date.today()
for stn in STNS:
    for y in range(START, END+1):
        t2 = f'{y}1231' if y<today.year else (today-timedelta(days=1)).strftime('%Y%m%d')
        # 빈 응답(0행)도 가끔 온다(2026-09-29 태백 2016·서귀포 2026) → 오류처럼 다시 요청
        for attempt in range(3):
            try:
                r=requests.get(URL,params={'tm1':f'{y}0101','tm2':t2,'stn':str(stn),
                                           'disp':0,'help':0,'authKey':key},timeout=120)
                if not any(l.strip() and not l.startswith('#') for l in r.text.splitlines()):
                    raise ValueError('빈 응답')
                break
            except Exception as e:
                print('retry',stn,y,type(e).__name__,file=sys.stderr); time.sleep(3)  # e 메시지엔 authKey 가 든 URL 이 섞인다
        else:
            continue
        n=0
        for line in r.text.splitlines():
            if line.startswith('#') or not line.strip(): continue
            f=line.split()
            if len(f)<40: continue
            def g(k):
                try: v=float(f[IDX[k]])
                except (ValueError,IndexError): return np.nan
                return np.nan if v in MISS else v
            rows.append(dict(tm=f[0],stn=int(f[1]),**{k:g(k) for k in IDX})); n+=1
        print(f'{STNS[stn]}({stn}) {y}: {n}행',file=sys.stderr)
        time.sleep(1.0)
df=pd.DataFrame(rows).drop_duplicates(['tm','stn'])
df.to_csv(OUT,index=False)
print('TOTAL',df.shape,df.tm.min(),df.tm.max(),file=sys.stderr)
print(df.groupby('stn').apply(lambda d: d.notna().mean().round(2).to_dict()).to_string()[:1500],file=sys.stderr)
