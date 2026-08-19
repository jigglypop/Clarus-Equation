from __future__ import annotations

import json
from pathlib import Path
import requests

PMCID='PMC11841214'
BASE='https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/supplmat.cgi/bioc_json/'
OUT=Path('md_loop_results'); OUT.mkdir(exist_ok=True)

s=requests.Session(); s.headers.update({'User-Agent':'Mozilla/5.0 scientific-reanalysis'})
list_url=f'{BASE}{PMCID}/list'
r=s.get(list_url,timeout=60); r.raise_for_status()
text=r.text
(OUT/'suppl_list_raw.txt').write_text(text,encoding='utf-8')
try:
    listing=r.json()
except Exception:
    listing={'raw':text}

# Fetch all indexed supplementary entries through the official API and retain parsed JSON/text.
entries=[]
for i in range(1,40):
    u=f'{BASE}{PMCID}/{i}'
    rr=s.get(u,timeout=60)
    if rr.status_code!=200 or not rr.text.strip():
        if i>20: break
        continue
    ctype=rr.headers.get('content-type','')
    try: payload=rr.json()
    except Exception: payload={'raw':rr.text[:200000]}
    blob=json.dumps(payload,ensure_ascii=False)
    # Keep entries relevant to source-data figs 3-5 or where filename metadata is missing but content names them.
    target=any(x in blob for x in ['source_data_Fig3','source_data_Fig4','source_data_Fig5','Fig3.xlsx','Fig4.xlsx','Fig5.xlsx'])
    entries.append({'index':i,'url':u,'content_type':ctype,'target':target,'payload':payload})

res={'pmcid':PMCID,'list_url':list_url,'listing':listing,'entries':entries}
(OUT/'inventory.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
lines=['# MD transthalamic source-data FAIR-SMART inventory','',f'PMCID `{PMCID}`','']
lines.append('## List response')
lines.append('```')
lines.append(text[:10000])
lines.append('```')
lines.append('')
lines.append('## Retrieved entries')
for e in entries:
    blob=json.dumps(e['payload'],ensure_ascii=False)
    lines.append(f'- index `{e["index"]}` target={e["target"]} chars={len(blob)} preview={blob[:500]!r}')
(OUT/'inventory.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print((OUT/'inventory.md').read_text())