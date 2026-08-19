from __future__ import annotations
import hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().with_name('route_dispositions.json')
IDS={'R-E17-F3','R-E17-F4','R-E17-F5','R-E17-F2','R-SLEEP-E19','R-SLEEP-E15','R-CELEGANS','R-GRID-TORUS','R-DANDI-37','R-ALLOPTICAL','R-BCI','R-MICRONS','R-SYNTH'}
LEGAL={'ELIGIBLE','PARTIAL_DESCRIPTIVE','UNTESTABLE_MISSING_INPUT','ACCESS_BLOCKED','INELIGIBLE_DEPENDENT','FAILED_EXECUTION'}
def walk(x):
 if isinstance(x,float): assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values(): walk(v)
 elif isinstance(x,list):
  for v in x: walk(v)
def main():
 d=json.loads(P.read_text());rs=d['routes'];assert {r['route_id'] for r in rs}==IDS and len(rs)==13
 assert all(r['status'] in LEGAL and r['reason'] for r in rs); assert d['confirmatory_family']==[]
 assert all(r['route_id']!='R-BCI' or r['status']=='PARTIAL_DESCRIPTIVE' for r in rs)
 route_text=' '.join(r['reason']+' '+json.dumps(r.get('details',{})) for r in rs).lower()
 assert 'proves' not in route_text and 'establishes delta' not in route_text
 for r in rs:
  for inp in r.get('details',{}).get('inputs',[]):
   p=Path(inp['path']); root=Path(__file__).resolve().parents[4]
   actual=root/p; assert actual.exists() and hashlib.sha256(actual.read_bytes()).hexdigest()==inp['sha256']
  walk(r)
 print('{"status":"PASS","routes":13}')
if __name__=='__main__':main()
