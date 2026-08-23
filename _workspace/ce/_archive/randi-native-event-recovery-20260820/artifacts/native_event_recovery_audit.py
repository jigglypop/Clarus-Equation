#!/usr/bin/env python3
"""Publication-native, outcome-blind event-input audit for Randi et al.

This program deliberately never requests or reads gcamp/response/effect data.
It uses only the standard library and makes no imports from the downloaded ZIP.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, re, shutil, stat, sys, time, urllib.error, urllib.parse, urllib.request, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OSF_CHILDREN = "https://api.osf.io/v2/files/671a5286badd54a2128707e3/"
ZENODO = "https://zenodo.org/api/records/8312985"
ALLOWED = ("ds_name.txt", "labels.txt", "stim_neurons.txt", "stim_volume_i.txt", "t.txt")
OUTCOME_WORDS = ("gcamp", "response", "autoresponse", "fluorescence", "deltaf", "dff", "kernel", "pvalue", "qvalue", "effect", "state", "fit")
MAX_FILES, MAX_OSF_BYTES, MAX_FAMILY_FILES, MAX_FAMILY_BYTES = 600, 25_000_000, 120, 15_000_000
ZIP_SIZE, ZIP_MD5 = 1_287_278, "40d87e790193d38528b4ba0cecf23e8c"
MAX_ZIP_MEMBERS, MAX_ZIP_UNCOMPRESSED = 10_000, 100_000_000
CONVERTER_COMMIT = "3544c9bb59f90d5630fa1871850d990db9cafc18"
CONVERTER_URL = "https://codeload.github.com/catalystneuro/leifer_lab_to_nwb/zip/" + CONVERTER_COMMIT
MAX_CONVERTER_BYTES = 5_000_000

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def digest(path, algorithm):
    h=hashlib.new(algorithm)
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def fetch(url, timeout=30, retries=3):
    req=urllib.request.Request(url, headers={"User-Agent":"Clarus-Equation-native-event-audit/1.0","Accept":"application/json"})
    last=None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return r.read(), dict(r.headers), r.url
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last=e
            if attempt+1 < retries: time.sleep(1 << attempt)
    raise RuntimeError("fetch failed after bounded retries: %s" % last)
def json_fetch(url): return json.loads(fetch(url)[0].decode("utf-8"))
def canonical(name, used):
    base=re.sub(r"[^A-Za-z0-9._-]+", "_", name.replace("/", "__")).strip("._") or "file"
    candidate=base; i=2
    while candidate.lower() in used: candidate="%s__%d"%(base,i); i+=1
    used.add(candidate.lower()); return candidate
def family(name):
    low=name.lower()
    return next((x for x in ALLOWED if low.endswith(x)), None)
def checksum_fields(attrs):
    raw=attrs.get("extra", {}).get("hashes") or attrs.get("hashes") or attrs.get("checksum")
    result=[]
    if isinstance(raw,dict): result=[{"algorithm":str(k).lower(),"value":str(v).lower()} for k,v in raw.items()]
    elif isinstance(raw,str):
        m=re.match(r"(?:(md5|sha256):)?([0-9a-fA-F]{32,64})$",raw)
        if m: result=[{"algorithm":m.group(1) or ("md5" if len(m.group(2))==32 else "sha256"),"value":m.group(2).lower()}]
    return [x for x in result if x["algorithm"] in ("md5","sha256")]
def osf_manifest():
    first=json_fetch(OSF_CHILDREN)
    first_data=first.get("data")
    if isinstance(first_data,dict):
        url=(((first_data.get("relationships") or {}).get("files") or {}).get("links") or {}).get("related",{}).get("href")
        if not url: raise RuntimeError("OSF folder record has no child listing relation")
    elif isinstance(first_data,list):
        url=OSF_CHILDREN
    else: raise RuntimeError("unexpected OSF folder response")
    rows=[]; seen=set()
    while url:
        if url in seen: raise RuntimeError("pagination cycle")
        seen.add(url); page=json_fetch(url); data=page.get("data",[])
        if not isinstance(data,list): raise RuntimeError("unexpected OSF child page shape")
        rows.extend(data); url=(page.get("links") or {}).get("next")
    normalized=[]
    for item in rows:
        a=item.get("attributes",{}); links=item.get("links",{}); name=a.get("name","")
        normalized.append({"id":item.get("id"),"name":name,"path":a.get("path"),"size":a.get("size"),"modified":a.get("modified"),"download_url":links.get("download"),"provider_checksums":checksum_fields(a),"kind":a.get("kind")})
    return sorted(normalized,key=lambda x:(x["path"] or "",x["id"] or "")), len(seen)
def select(rows):
    candidates=[]
    for r in rows:
        n=(r["name"] or "").lower(); f=family(n)
        if f and not any(w in n for w in OUTCOME_WORDS): candidates.append(dict(r,family=f))
    candidates.sort(key=lambda r:(ALLOWED.index(r["family"]), r["size"] if isinstance(r["size"],int) else 10**99, r["path"] or "", r["id"] or ""))
    counts=Counter(); sizes=Counter(); selected=[]
    for r in candidates:
        size=r["size"]
        if not isinstance(size,int) or size<0: raise RuntimeError("missing/invalid declared size: %s"%r["id"])
        if len(selected)+1>MAX_FILES or sum(sizes.values())+size>MAX_OSF_BYTES or counts[r["family"]]+1>MAX_FAMILY_FILES or sizes[r["family"]]+size>MAX_FAMILY_BYTES:
            raise RuntimeError("BLOCKED_ACQUISITION_BOUND before download")
        selected.append(r); counts[r["family"]]+=1; sizes[r["family"]]+=size
    return selected, dict(counts), dict(sizes)
def download(url, out, expected_size=None):
    data, _, final=fetch(url, timeout=60, retries=3)
    if expected_size is not None and len(data)!=expected_size: raise RuntimeError("size mismatch %s != %s"%(len(data),expected_size))
    out.write_bytes(data); return final
def read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8",errors="strict").splitlines() if line.strip()]
def raw_lines(path):
    # labels.txt is an index-addressed table: blank rows are meaningful placeholders.
    return [line.strip() for line in path.read_text(encoding="utf-8",errors="strict").splitlines()]
def numeric_rows(path):
    vals=[]
    for line in read_lines(path):
        tokens=re.split(r"[\s,;]+",line)
        row=[]
        for t in tokens:
            try: row.append(float(t))
            except ValueError: raise ValueError("non-primitive numeric token in %s: %r"%(path.name,t))
        vals.append(row)
    return vals
def prefix(name, suffix):
    return name[: -len(suffix)] if name.lower().endswith(suffix) else None
def zip_audit(zpath):
    result={"member_count":0,"uncompressed_bytes":0,"unsafe_members":[],"source_files_scanned":[],"native_field_loading_paths":[],"assignment_semantics_hits":{}}
    with zipfile.ZipFile(zpath) as z:
        infos=z.infolist(); result["member_count"]=len(infos); result["uncompressed_bytes"]=sum(i.file_size for i in infos)
        if len(infos)>MAX_ZIP_MEMBERS or result["uncompressed_bytes"]>MAX_ZIP_UNCOMPRESSED: raise RuntimeError("unsafe ZIP aggregate bounds")
        for i in infos:
            p=Path(i.filename)
            unsafe=(p.is_absolute() or ".." in p.parts or bool(i.flag_bits&1) or stat.S_ISLNK(i.external_attr>>16))
            if unsafe: result["unsafe_members"].append(i.filename)
        if result["unsafe_members"]: raise RuntimeError("unsafe ZIP member metadata")
        # Static source inspection only: do not extract, import, or execute.
        for i in infos:
            low=i.filename.lower()
            if not low.endswith((".py",".md",".rst",".txt")) or i.file_size>2_000_000: continue
            if any(w in low for w in ("gcamp","response","fluorescence","kernel","qvalue","pvalue")): continue
            text=z.read(i).decode("utf-8",errors="replace")
            if low.endswith(".py"):
                result["source_files_scanned"].append(i.filename)
                for line_no,line in enumerate(text.splitlines(),1):
                    l=line.lower()
                    if any(k in l for k in ("stim_neurons","stim_volume_i","ds_name","labels", "_t.txt")):
                        result["native_field_loading_paths"].append({"path":i.filename,"line":line_no,"text":line.strip()[:300]})
                    for term in ("automatic","manual","fail","assignment","exclude"):
                        if term in l: result["assignment_semantics_hits"].setdefault(term,[]).append({"path":i.filename,"line":line_no,"text":line.strip()[:300]})
    return result
def converter_audit(zpath):
    """Read converter ZIP metadata/source only; no extraction, import, or execution."""
    terms=("stim_neurons", "stim_volume_i", "labels", "ds_name", "manual", "fail", "-1", "-2", "-3", "TargetPlaneSegmentation", "NeuroPAL", "PlaneSegmentation", "target")
    result={"commit":CONVERTER_COMMIT,"member_count":0,"uncompressed_bytes":0,"unsafe_members":[],"source_files_scanned":[],"term_hits":{t:[] for t in terms}}
    with zipfile.ZipFile(zpath) as z:
        infos=z.infolist(); result["member_count"]=len(infos); result["uncompressed_bytes"]=sum(i.file_size for i in infos)
        if len(infos)>MAX_ZIP_MEMBERS or result["uncompressed_bytes"]>MAX_ZIP_UNCOMPRESSED: raise RuntimeError("unsafe converter ZIP aggregate bounds")
        prefix="leifer_lab_to_nwb-"+CONVERTER_COMMIT
        if not any(i.filename.startswith(prefix+"/") for i in infos): raise RuntimeError("converter archive root does not match frozen commit")
        for i in infos:
            p=Path(i.filename); unsafe=(p.is_absolute() or ".." in p.parts or bool(i.flag_bits&1) or stat.S_ISLNK(i.external_attr>>16))
            if unsafe: result["unsafe_members"].append(i.filename)
        if result["unsafe_members"]: raise RuntimeError("unsafe converter ZIP member metadata")
        for i in infos:
            low=i.filename.lower()
            if not low.endswith((".py",".yaml",".yml",".json",".md",".rst",".txt")) or i.file_size>2_000_000: continue
            text=z.read(i).decode("utf-8",errors="replace")
            result["source_files_scanned"].append(i.filename)
            for n,line in enumerate(text.splitlines(),1):
                for term in terms:
                    if term.lower() in line.lower(): result["term_hits"][term].append({"path":i.filename,"line":n,"text":line.strip()[:300]})
    # Presence is only static schema evidence; it is deliberately not an event-row receipt.
    result["static_mapping_status"]={
        "native_stim_fields_seen":bool(result["term_hits"]["stim_neurons"] or result["term_hits"]["stim_volume_i"]),
        "manual_terms_seen":bool(result["term_hits"]["manual"]),
        "failure_or_sentinel_terms_seen":bool(result["term_hits"]["fail"] or result["term_hits"]["-1"] or result["term_hits"]["-2"] or result["term_hits"]["-3"]),
        "targetplanesegmentation_seen":bool(result["term_hits"]["TargetPlaneSegmentation"]),
        "neuropal_seen":bool(result["term_hits"]["NeuroPAL"]),
        "assignment_receipt_preserved":False,
        "canonical_identity_provenance_preserved":False,
    }
    opto="src/leifer_lab_to_nwb/randi_nature_2023/interfaces/_optogenetic_stimulation.py"
    opto_hits=[h for group in result["term_hits"].values() for h in group if opto in h["path"]]
    result["field_mapping_audit"]={
        "native_stim_neurons_or_volume_field": "NOT_FOUND_IN_CONVERTER_SOURCE",
        "native_sentinels_minus_1_minus_2_minus_3": "NOT_FOUND_AS_EVENT_SENTINEL_MAPPING",
        "complementary_label_confidence_comment_arrays": "NEUROPAL_LABELS_CONFIDENCES_COMMENTS_STATIC_SCHEMA_SEEN" if result["term_hits"]["labels"] else "NOT_FOUND",
        "manual_target_input": "targets_manually_located.txt" if any("targets_manually_located.txt" in h["text"] for h in opto_hits) else "NOT_FOUND",
        "manual_target_output": "STATIC_MANUALLY_TARGETED_ID_SEEN" if any("Manually targeted ID" in h["text"] for h in opto_hits) else "NOT_FOUND",
        "failed_target_representation": "CAST_TO_NAN_STATICALLY_SEEN" if any("Cast to NaN" in h["text"] for h in opto_hits) else "NOT_FOUND",
        "target_plane_segmentation": "STATIC_SEEN" if result["static_mapping_status"]["targetplanesegmentation_seen"] else "NOT_FOUND",
        "neuropal_mapping": "STATIC_SEEN" if result["static_mapping_status"]["neuropal_seen"] else "NOT_FOUND",
        "conclusion": "CONVERTER_STATIC_SCHEMA_ONLY__NO_COMPLETE_NATIVE_EVENT_ASSIGNMENT_OR_CANONICAL_PROVENANCE_RECEIPT",
    }
    return result
def data_audit(files, root):
    by_session=defaultdict(dict)
    for f in files:
        p=root/Path(f["local_path"]); pre=prefix(p.name,f["family"])
        if pre is None: raise RuntimeError("canonical filename lost suffix")
        by_session[pre][f["family"]]=p
    sessions={}; all_complete=True
    for sid, fs in sorted(by_session.items()):
        missing=[x for x in ALLOWED if x not in fs]; complete=not missing; all_complete &= complete
        rec={"families":sorted(fs),"missing_families":missing,"complete":complete}
        if complete:
            labels=raw_lines(fs["labels.txt"]); stim=numeric_rows(fs["stim_neurons.txt"]); vol=numeric_rows(fs["stim_volume_i.txt"]); clock=numeric_rows(fs["t.txt"])
            ds=read_lines(fs["ds_name.txt"])
            idx=[r[0] for r in stim if r]
            integral=all(math.isfinite(x) and x.is_integer() for x in idx)
            sentinels=sorted({int(x) for x in idx if x<0}) if integral else []
            assigned=[int(x) for x in idx if x>=0] if integral else []
            valid=integral and all(x<len(labels) for x in assigned)
            join=[labels[x] for x in assigned] if valid else []
            labeled_join=[x for x in join if x]
            monotonic = bool(clock) and all(clock[i][0] >= clock[i - 1][0] for i in range(1, len(clock)))
            rec.update({"ds_name_rows":len(ds),"labels_rows":len(labels),"stim_neurons_rows":len(stim),"stim_volume_i_rows":len(vol),"clock_rows":len(clock),"event_cardinality_equal":len(stim)==len(vol),"stim_index_integral":integral,"stim_index_in_labels_domain":valid,"labels_stim_neurons_join_count":len(join),"labels_stim_neurons_nonblank_join_count":len(labeled_join),"labels_stim_neurons_blank_join_count":len(join)-len(labeled_join),"stim_sentinels":sentinels,"clock_nonempty":bool(clock),"clock_monotonic_first_column":monotonic})
        sessions[sid]=rec
    return {"session_count":len(sessions),"expected_session_count":113,"all_allowed_family_complete":all_complete,"numeric_prefix_alignment":all(len(v["families"])<=len(ALLOWED) for v in sessions.values()),"sessions":sessions}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--offline",action="store_true"); ap.add_argument("--fetch-converter",action="store_true"); args=ap.parse_args()
    root=Path(args.run_dir).resolve(); art=root/"artifacts"; native=art/"native_files"; art.mkdir(parents=True,exist_ok=True); native.mkdir(parents=True,exist_ok=True)
    manifest_path=art/"osf_raw_extracted_manifest.json"; result={"contract":"outcome_blind_native_event_recovery","forbidden_data_read":False,"status":[]}
    if args.offline:
        rows=json.loads(manifest_path.read_text())["objects"]
        pages=None
    else:
        rows,pages=osf_manifest(); manifest_path.write_text(json.dumps({"api":OSF_CHILDREN,"metadata_pages":pages,"objects":rows},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    selected,counts,sizes=select(rows); used=set(); receipts=[]
    for r in selected:
        out=native/canonical(r["name"],used); r["local_path"]=str(out.relative_to(root)).replace("\\","/")
    def acquire(r):
        out=root/r["local_path"]
        if out.exists() and out.stat().st_size == r["size"]: return "RESUMED_LOCAL_SIZE_MATCH"
        return download(r["download_url"],out,r["size"])
    if not args.offline:
        # Bounded concurrency remains noninteractive and avoids an unbounded request fan-out.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for r, final in zip(selected, pool.map(acquire, selected)): r["download_final_url"] = final
    for r in selected:
        out=root/r["local_path"]
        if not out.exists(): raise RuntimeError("missing selected download: "+str(out))
        verified=[]
        for c in r["provider_checksums"]: verified.append({**c,"match":digest(out,c["algorithm"])==c["value"]})
        r["provider_checksum_present"]=bool(verified); r["provider_checksum_verified"]=bool(verified) and all(x["match"] for x in verified); r["provider_checksum_results"]=verified; r["download_sha256"]=sha256(out); receipts.append(r)
    (art/"selected_native_files.json").write_text(json.dumps({"selection_order":"family priority, byte size, canonical path","family_counts":counts,"family_bytes":sizes,"files":receipts},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    zpath=art/"pumpprobe-1.1.zip"
    if not args.offline:
        record=json_fetch(ZENODO); zfile=next((x for x in record.get("files",[]) if x.get("key")=="pumpprobe-1.1.zip"),None)
        if not zfile: raise RuntimeError("Zenodo target file absent")
        if zfile.get("size")!=ZIP_SIZE or (zfile.get("checksum") or "").lower()!="md5:"+ZIP_MD5: raise RuntimeError("Zenodo metadata receipt mismatch")
        if not (zpath.exists() and zpath.stat().st_size == ZIP_SIZE): download(zfile["links"]["self"],zpath,ZIP_SIZE)
    if zpath.stat().st_size!=ZIP_SIZE or digest(zpath,"md5")!=ZIP_MD5: raise RuntimeError("Zenodo bytes receipt mismatch")
    cpath=art/("leifer_lab_to_nwb-"+CONVERTER_COMMIT+".zip")
    if args.fetch_converter and not (cpath.exists() and cpath.stat().st_size <= MAX_CONVERTER_BYTES):
        final=download(CONVERTER_URL,cpath)
    else: final="LOCAL_ONLY" if cpath.exists() else None
    if not cpath.exists() or cpath.stat().st_size>MAX_CONVERTER_BYTES: raise RuntimeError("converter archive missing or over 5 MB cap")
    total_files=len(receipts)+2; total_bytes=sum(x["size"] for x in receipts)+ZIP_SIZE+cpath.stat().st_size
    if total_files>602 or total_bytes>32_000_000: raise RuntimeError("BLOCKED_ACQUISITION_BOUND global R1b cap")
    za=zip_audit(zpath); ca=converter_audit(cpath); da=data_audit(receipts, root)
    result.update({"osf_metadata_pages":pages,"selected_file_count":len(receipts),"family_counts":counts,"manifest_sha256":sha256(manifest_path),"zenodo":{"file":"pumpprobe-1.1.zip","bytes":ZIP_SIZE,"md5":ZIP_MD5,"local_sha256":sha256(zpath)},"converter":{"repository":"catalystneuro/leifer_lab_to_nwb","commit":CONVERTER_COMMIT,"request_url":CONVERTER_URL,"final_url":final,"bytes":cpath.stat().st_size,"local_sha256":sha256(cpath)},"global_acquisition":{"file_count":total_files,"byte_count":total_bytes,"file_cap":602,"byte_cap":32_000_000},"zip_static_audit":za,"converter_static_audit":ca,"native_data_audit":da})
    index_pass=da["all_allowed_family_complete"] and da["session_count"]==113 and all(x.get("stim_index_in_labels_domain") and x.get("event_cardinality_equal") and x.get("clock_nonempty") for x in da["sessions"].values())
    result["gate_status"]={"PASS_SOURCE_INDEX_JOIN":index_pass,"PASS_SOURCE_JOIN":False,"PASS_ASSIGNMENT_RECEIPT":False,"PASS_APPARATUS_INPUT":False,"canonical_identity_confidence_provenance":"NOT_PRESENT_IN_ALLOWED_NATIVE_FILES","assignment_receipt":"NOT_ESTABLISHED_BY_STATIC_SOURCE_OR_ALLOWED_NATIVE_FILES","overall":"PASS_OBSERVATIONAL_ONLY" if index_pass else "BLOCKED_SOURCE_INDEX_JOIN"}
    (art/"native_event_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"overall":result["gate_status"]["overall"],"source_index_join":index_pass,"files":len(receipts),"sessions":da["session_count"]},sort_keys=True))
if __name__=="__main__": main()
