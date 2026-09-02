import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
def walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, pre + "/" + str(k))
    elif isinstance(o, list) and len(o) > 8:
        print(pre, "list len", len(o))
    else:
        print(pre, "=", o)
walk(d)
