"""a6: (a) verify the Cayley constants behind recovers[3] (the +1.279% b4 recovery);
       (b) large-n behaviour of the amplitude factor S_gen / D  -- the card only checks the
           i.i.d. limit S_gen/D -> 1 (verify[13]); for the heritable families that motivated
           F-02 the same ratio decides whether "amplitude non-universality" survives n -> infinity.
"""
import json, math, sys
from itertools import product
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import caterpillar, ancestor_matrix

OUT = Path(__file__).parent


def sub_sizes(parent):
    n = len(parent)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    order, i = [root], 0
    while i < len(order):
        order.extend(ch[order[i]])
        i += 1
    s = np.ones(n, dtype=np.int64)
    for v in reversed(order[1:]):
        s[parent[v]] += s[v]
    return s


def S_gen_from_sizes(s, n):
    return float(np.sum((s * (1.0 - s / n)) ** 2))


def D_exact(parent):
    n = len(parent)
    A = ancestor_matrix(parent)
    H = np.eye(n) - np.ones((n, n)) / n
    B = A.T @ H @ A
    return float(np.trace(B @ B)), float(np.sum(np.diag(B) ** 2))


def meir_moon_S(n):
    """E S_gen = sum_k N_k k^2 (1-k/n)^2 with N_k = C(n,k) k^{k-1} (n-k)^{n-k} / n^{n-1}."""
    total = 0.0
    for k in range(1, n + 1):
        lnN = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
               + (k - 1) * math.log(k) + ((n - k) * math.log(n - k) if n - k > 0 else 0.0)
               - (n - 1) * math.log(n))
        total += math.exp(lnN) * k * k * (1 - k / n) ** 2
    return total


def enumerate_rooted_trees(n):
    """all n^{n-1} rooted labelled trees on {0..n-1} with root 0 (parent function coding)."""
    for par in product(range(n), repeat=n - 1):
        parent = [-1] + list(par)
        seen = True
        for v in range(1, n):
            u, steps = v, 0
            while u > 0 and steps <= n:
                u = parent[u]
                steps += 1
            if u != 0:
                seen = False
                break
        if seen:
            yield parent


def random_cayley(n, rng):
    """uniform rooted labelled tree via Prufer + random root."""
    import heapq
    if n <= 2:
        return [-1] + [0] * (n - 1)
    seq = rng.integers(0, n, size=n - 2)
    deg = np.ones(n, dtype=int)
    for s in seq:
        deg[s] += 1
    adj = [[] for _ in range(n)]
    leaves = [i for i in range(n) if deg[i] == 1]
    heapq.heapify(leaves)
    for s in seq:
        lf = heapq.heappop(leaves)
        adj[lf].append(int(s))
        adj[int(s)].append(lf)
        deg[s] -= 1
        if deg[s] == 1:
            heapq.heappush(leaves, int(s))
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    adj[u].append(v)
    adj[v].append(u)
    root = int(rng.integers(0, n))
    parent = [-2] * n
    parent[root] = -1
    st = [root]
    while st:
        x = st.pop()
        for y in adj[x]:
            if parent[y] == -2:
                parent[y] = x
                st.append(y)
    return parent


def cayley_D_S(parent):
    """O(n) exact D = tr(B^2) and S_gen for the ancestor generator (B_uw = |sub(u) cap sub(w)| - s_u s_w / n
    -> use the F-02 driver identity D = W2' - 2 S_row / n + W2^2 / n^2)."""
    n = len(parent)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    order, i = [root], 0
    while i < len(order):
        order.extend(ch[order[i]])
        i += 1
    depth = np.zeros(n, dtype=np.int64)
    s = np.ones(n, dtype=np.int64)
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1
    for v in reversed(order[1:]):
        s[parent[v]] += s[v]
    pre = np.zeros(n, dtype=np.int64)
    for v in order:
        pre[v] = s[v] + (pre[parent[v]] if parent[v] >= 0 else 0)
    sf = s.astype(float)
    w2 = float(np.sum(sf * sf))
    w2p = float(np.sum((2 * depth + 1) * sf * sf))
    srow = float(np.sum(pre.astype(float) ** 2))
    D = w2p - 2 * srow / n + w2 * w2 / (n * n)
    S = float(np.sum((sf * (1 - sf / n)) ** 2))
    return D, S


def main():
    res = {}
    # (0) exhaustive check of Meir-Moon E S_gen for small n
    ex = {}
    for n in (3, 4, 5, 6, 7):
        tot, cnt = 0.0, 0
        for parent in enumerate_rooted_trees(n):
            tot += S_gen_from_sizes(sub_sizes(parent), n)
            cnt += 1
        ex[n] = {"trees": cnt, "exhaustive": tot / cnt, "meir_moon": meir_moon_S(n)}
        print("(0) n=%d  trees=%6d  exhaustive E S_gen = %.10f   Meir-Moon = %.10f   diff %.2e"
              % (n, cnt, ex[n]["exhaustive"], ex[n]["meir_moon"],
                 abs(ex[n]["exhaustive"] - ex[n]["meir_moon"])))
    res["exhaustive_vs_meir_moon"] = ex

    # (1) Cayley n = 32: the numbers the card uses in recovers[3] / verify[12]
    rng = np.random.default_rng(20260902 + 31337)
    cay = {}
    for n in (32, 64, 128, 256, 512, 1024):
        reps = 40000 if n <= 128 else (12000 if n <= 512 else 4000)
        Ds = np.empty(reps)
        Ss = np.empty(reps)
        for t in range(reps):
            Ds[t], Ss[t] = cayley_D_S(random_cayley(n, rng))
        cay[n] = {"reps": reps, "E_D": float(Ds.mean()), "se_D": float(Ds.std(ddof=1) / math.sqrt(reps)),
                  "E_S_gen": float(Ss.mean()), "se_S": float(Ss.std(ddof=1) / math.sqrt(reps)),
                  "meir_moon_S": meir_moon_S(n), "ratio_ES_over_ED": float(Ss.mean() / Ds.mean())}
        print("(1) Cayley n=%4d  E D = %12.3f +-%8.3f   E S_gen = %11.3f +-%7.3f  (Meir-Moon %11.3f)"
              "   E S/E D = %.5f" % (n, cay[n]["E_D"], cay[n]["se_D"], cay[n]["E_S_gen"],
                                     cay[n]["se_S"], cay[n]["meir_moon_S"], cay[n]["ratio_ES_over_ED"]))
    res["cayley"] = cay
    card = {"E_S_gen_32": 444.50685, "E_D_32": 2008.0806, "ratio": 0.221359066}
    res["card_cayley32"] = card
    z = (cay[32]["E_S_gen"] - card["E_S_gen_32"]) / cay[32]["se_S"]
    print("    card E S_gen(32) = %.5f  vs MC %.3f+-%.3f (z=%+.2f)  vs Meir-Moon %.5f"
          % (card["E_S_gen_32"], cay[32]["E_S_gen"], cay[32]["se_S"], z, meir_moon_S(32)))

    # (2) caterpillar family: exact S_gen / D as k grows (n = k^2)
    cat = {}
    for k in (3, 4, 6, 8, 10, 14, 20, 28, 40):
        parent = caterpillar(k)
        D, S = D_exact(parent)
        cat[k] = {"n": k * k, "D": D, "S_gen": S, "ratio": S / D,
                  "a": S / D / 60.0, "rho_spike64": 1 + 61 * S / D / 60.0}
        print("(2) caterpillar k=%2d (n=%4d)  S_gen/D = %.6f   a = %.6f   rho(spike64) = %.5f"
              % (k, k * k, S / D, S / D / 60.0, 1 + 61 * S / D / 60.0))
    res["caterpillar"] = cat
    ks = sorted(cat)
    ratios = [cat[k]["ratio"] for k in ks]
    ns = [cat[k]["n"] for k in ks]
    slope = float(np.polyfit(np.log(ns[-4:]), np.log(ratios[-4:]), 1)[0])
    res["caterpillar_ratio_loglog_slope_large_k"] = slope
    print("    caterpillar S_gen/D  log-log slope vs n (largest four) = %+.4f" % slope)
    nsc = sorted(cay)
    slope_c = float(np.polyfit(np.log(nsc[-4:]), np.log([cay[n]["ratio_ES_over_ED"] for n in nsc[-4:]]), 1)[0])
    res["cayley_ratio_loglog_slope"] = slope_c
    print("    Cayley  E S_gen / E D  log-log slope vs n (largest four) = %+.4f" % slope_c)
    (OUT / "a6_asymptotics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
