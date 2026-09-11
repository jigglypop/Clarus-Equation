from __future__ import annotations

import io, json, zipfile
from pathlib import Path
import pandas as pd, requests

PMCID='PMC11841214'
URL=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles'
OUT=Path('md_loop_results'); OUT.mkdir(exist_ok=True)
s=requests.Session(); s.headers.update({'User-Agent':'Mozilla/5.0 scientific-reanalysis'})
r=s.get(URL,timeout=120); r.raise_for_status()
raw=r.content
(OUT/'supplementary.zip').write_bytes(raw)
res={'pmcid':PMCID,'url':URL,'bytes':len(raw),'members':[]}
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        info=z.getinfo(name)
        item={'name':name,'bytes':info.file_size}
        if name.lower().endswith('.xlsx') and any(k in name for k in ['source_data_Fig3','source_data_Fig4','source_data_Fig5']):
            data=z.read(name); p=OUT/Path(name).name; p.write_bytes(data)
            xls=pd.ExcelFile(io.BytesIO(data),engine='openpyxl')
            sheets={}
            for sh in xls.sheet_names:
                df=pd.read_excel(io.BytesIO(data),sheet_name=sh,header=None,engine='openpyxl')
                ne=df.dropna(how='all').dropna(axis=1,how='all')
                preview=ne.head(25).astype(object).where(pd.notna(ne),None).values.tolist()
                sheets[sh]={'shape':list(ne.shape),'preview':preview}
            item['target']=True; item['sheets']=sheets
        res['members'].append(item)
(OUT/'inventory.json').write_text(json.dumps(res,indent=2,default=str),encoding='utf-8')
lines=['# MD transthalamic Europe PMC supplementary inventory','',f'URL `{URL}` bytes={len(raw)}','']
for m in res['members']:
    if m.get('target'):
        lines.append(f'## {m["name"]}')
        for sh,meta in m['sheets'].items():
            lines.append(f'- `{sh}` shape={meta["shape"]}')
            for row in meta['preview'][:10]: lines.append('  - '+repr(row[:10]))
(OUT/'inventory.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print((OUT/'inventory.md').read_text())