from __future__ import annotations

import json, re
from pathlib import Path
import pandas as pd, requests

BASE='https://pmc.ncbi.nlm.nih.gov/articles/PMC11841214/bin/'
FILES={
 'fig3':'NIHMS2040407-supplement-source_data_Fig3.xlsx',
 'fig4':'NIHMS2040407-supplement-source_data_Fig4.xlsx',
 'fig5':'NIHMS2040407-supplement-source_data_Fig5.xlsx',
}
OUT=Path('md_loop_results'); OUT.mkdir(exist_ok=True)
res={}
for key,name in FILES.items():
    url=BASE+name
    data=requests.get(url,timeout=60).content
    p=OUT/name; p.write_bytes(data)
    xls=pd.ExcelFile(p)
    sheets={}
    for sh in xls.sheet_names:
        df=pd.read_excel(p,sheet_name=sh,header=None)
        nonempty=df.dropna(how='all').dropna(axis=1,how='all')
        preview=nonempty.head(20).astype(object).where(pd.notna(nonempty),None).values.tolist()
        sheets[sh]={'shape':list(nonempty.shape),'preview':preview}
    res[key]={'url':url,'bytes':len(data),'sheets':sheets}
(OUT/'inventory.json').write_text(json.dumps(res,indent=2,default=str),encoding='utf-8')
lines=['# MD transthalamic source-data inventory','']
for k,v in res.items():
    lines.append(f'## {k.upper()}')
    for sh,meta in v['sheets'].items():
        lines.append(f'- `{sh}` shape={meta["shape"]}')
        for row in meta['preview'][:6]: lines.append('  - '+repr(row[:8]))
(OUT/'inventory.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print((OUT/'inventory.md').read_text())