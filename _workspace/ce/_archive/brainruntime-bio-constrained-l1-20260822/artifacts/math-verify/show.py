import json, os, sys
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
d = json.load(open(p))
keys = sys.argv[2:] if len(sys.argv) > 2 else list(d)
for k in keys:
    print("== " + k)
    print(json.dumps(d[k], indent=1))
