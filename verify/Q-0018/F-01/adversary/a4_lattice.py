"""Adversary a4: lattice constants, Cayley combinatorics, window discrimination, dof."""
from __future__ import annotations
import itertools, json, math, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from driver_numbers import cayley_exact, n_k_log

C2, C4 = Fraction(640, 81), Fraction(-352, 81)
out = {"script": "a4_lattice"}


def centering(n):
    return np.eye(n) - np.ones((n, n)) / n


def gen(name, n):
    if name == "iid":
        return np.eye(n)
    if name == "chain":
        return np.tril(np.ones((n, n)))
    A = np.zeros((n, 2))
    A[:n // 2, 0] = 1.0
    A[n // 2:, 1] = 1.0
    return A


lat = {}
for name in ("iid", "chain", "coh"):
    for n in (2, 4, 8, 16):
        A = gen(name, n)
        B = A.T @ centering(n) @ A
        K = centering(n) @ A @ A.T @ centering(n)
        D = float(np.sum(B * B))
        S = float(np.sum(np.diag(B) ** 2))
        Sker = float(np.sum(np.diag(K) ** 2))
        if name == "iid":
            eD, eS = Fraction(n - 1), Fraction((n - 1) ** 2, n)
        elif name == "chain":
            eD = Fraction((n * n - 1) * (2 * n * n + 7), 180)
            eS = sum(Fraction(k * k) * (1 - Fraction(k, n)) ** 2 for k in range(1, n + 1))
        else:
            eD, eS = Fraction(n * n, 4), Fraction(n * n, 8)
        cn = C2 + C4 * eS / eD
        lat[name + "_n" + str(n)] = {
            "D_matrix": D, "D_closed": str(eD), "D_ok": abs(D - float(eD)) < 1e-9,
            "S_matrix": S, "S_closed": str(eS), "S_ok": abs(S - float(eS)) < 1e-9,
            "S_over_D": float(eS / eD), "c_pred": float(cn), "c_pred_exact": str(cn),
            "S_ker": Sker, "c_alt_kernel_diag": float(C2) + float(C4) * Sker / float(eD),
            "c_alt_C_is_I16": 10.0 + (32 / 9 - 10) * float(eS / eD)}
out["lattice"] = lat
c_text = float(C2 + C4 * Fraction(1, 2)) / 4.0
out["P4_denominator_audit"] = {
    "card_text_denominator": 256.0, "true_D_coh_16": lat["coh_n16"]["D_matrix"],
    "F02_closed_form_4n2p2q2": 4 * 16 ** 2 * 0.25 * 0.25, "card_result_json_D": 64.0,
    "factor_error_if_text_followed": 4.0,
    "c_that_the_card_text_would_produce": c_text,
    "K3_window": [5.4420, 6.0148],
    "K3_would_falsely_fire": not (5.4420 <= c_text <= 6.0148)}
S16 = sum(Fraction(k * k) * (1 - Fraction(k, 16)) ** 2 for k in range(1, 17))
D16 = Fraction((16 * 16 - 1) * (2 * 16 * 16 + 7), 180)
R = (Fraction(15, 16) - S16 / D16) / (Fraction(15, 16) - Fraction(1, 2))
out["R_str"] = {"exact": str(R), "float": float(R), "card": 1.718414533443435,
                "chain_D16_exact": str(D16), "chain_S16_exact": str(S16),
                "S_over_D_16": float(S16 / D16),
                "S_over_D_matches_257_1384": S16 / D16 == Fraction(257, 1384)}


def stats_from_adj(adj, root, n):
    dep = [0] * n
    parent = [-1] * n
    stack = [root]
    seen = [False] * n
    seen[root] = True
    order = []
    while stack:
        v = stack.pop()
        order.append(v)
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                parent[w] = v
                dep[w] = dep[v] + 1
                stack.append(w)
    size = [1] * n
    for v in reversed(order):
        if parent[v] >= 0:
            size[parent[v]] += size[v]
    A = np.zeros((n, n))
    for v in range(n):
        u = v
        while u >= 0:
            A[v, u] = 1.0
            u = parent[u]
    B = A.T @ centering(n) @ A
    D = float(np.sum(B * B))
    S = float(np.sum(np.diag(B) ** 2))
    S2 = float(sum(s * s * (1 - s / n) ** 2 for s in size))
    return D, S, S2


def adj_from_prufer(seq, n):
    import heapq
    deg = [1] * n
    for s in seq:
        deg[s] += 1
    adj = [[] for _ in range(n)]
    leaves = [i for i in range(n) if deg[i] == 1]
    heapq.heapify(leaves)
    for s in seq:
        lf = heapq.heappop(leaves)
        adj[lf].append(s)
        adj[s].append(lf)
        deg[s] -= 1
        if deg[s] == 1:
            heapq.heappush(leaves, s)
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    adj[u].append(v)
    adj[v].append(u)
    return adj


def meir_moon_S(n):
    return sum(math.exp(n_k_log(n, k)) * k * k * (1 - k / n) ** 2 for k in range(1, n + 1))


cay = {}
for n in (4, 5, 6):
    tot_d = tot_s = 0.0
    cnt = 0
    for seq in itertools.product(range(n), repeat=n - 2):
        adj = adj_from_prufer(list(seq), n)
        for root in range(n):
            D, S, S2 = stats_from_adj(adj, root, n)
            assert abs(S - S2) < 1e-9
            tot_d += D
            tot_s += S
            cnt += 1
    cay["brute_n" + str(n)] = {"trees": cnt, "E_D_brute": tot_d / cnt,
                               "E_D_driver": cayley_exact(n)["E_D"],
                               "E_S_brute": tot_s / cnt, "E_S_meir_moon": meir_moon_S(n),
                               "S_rel_err": tot_s / cnt / meir_moon_S(n) - 1}
rng = np.random.default_rng(20260903)
for n, T in ((32, 4000), (128, 3000)):
    ds, ss = [], []
    for _ in range(T):
        seq = list(rng.integers(0, n, size=n - 2))
        adj = adj_from_prufer(seq, n)
        D, S, _ = stats_from_adj(adj, int(rng.integers(0, n)), n)
        ds.append(D)
        ss.append(S)
    ds = np.array(ds)
    ss = np.array(ss)
    cay["mc_n" + str(n)] = {"trials": T, "E_D_mc": float(ds.mean()),
                            "E_D_driver": cayley_exact(n)["E_D"],
                            "E_S_mc": float(ss.mean()), "E_S_meir_moon": meir_moon_S(n),
                            "S_z": float((ss.mean() - meir_moon_S(n)) / (ss.std(ddof=1) / math.sqrt(T))),
                            "D_z": float((ds.mean() - cayley_exact(n)["E_D"]) / (ds.std(ddof=1) / math.sqrt(T)))}
out["cayley"] = cay

grid = (8, 16, 32, 64, 128)
c2f, c4f = float(C2), float(C4)
ED = {n: cayley_exact(n)["E_D"] for n in grid}
ES = {n: meir_moon_S(n) for n in grid}
rms_her = [math.sqrt(c2f * ED[n] + c4f * ES[n]) / n for n in grid]
rms_iid = [math.sqrt((c2f + c4f * (n - 1) / n) * (n - 1)) / n for n in grid]
rms_her_g = [math.sqrt(10 * ED[n]) / n for n in grid]
rms_iid_g = [math.sqrt(10 * (n - 1)) / n for n in grid]


def sl(ys):
    return float(np.polyfit(np.log(grid), np.log(ys), 1)[0])


ratio_det = rms_her[-1] / rms_iid[-1]
ratio_g = rms_her_g[-1] / rms_iid_g[-1]
out["K5_numbers"] = {
    "E_S_gen_cayley": {str(n): ES[n] for n in grid},
    "E_D_cayley": {str(n): ED[n] for n in grid},
    "gamma_her_det": sl(rms_her), "card_gamma_her": 0.5717833946085775,
    "gamma_iid_det": sl(rms_iid), "card_gamma_iid": -0.5014333477295538,
    "gamma_her_gauss": sl(rms_her_g), "gamma_iid_gauss": sl(rms_iid_g),
    "ratio_128_det": ratio_det, "card_ratio": 46.8504,
    "ratio_128_gauss": ratio_g, "card_ratio_gauss": 32.554,
    "K5_window": [37.48, 56.22], "gauss_inside_K5_window": bool(37.48 <= ratio_g <= 56.22),
    "F02_K1_window": [26.0, 39.1], "det_inside_F02_K1_window": bool(26.0 <= ratio_det <= 39.1),
    "window_overlap": [37.48, 39.1],
    "overlap_fraction_of_K5_window": (39.1 - 37.48) / (56.22 - 37.48),
    "F02_K1_stated_uncertainty_band": [32.554 - 6.5, 32.554 + 6.5],
    "K5_slope_window": [0.47, 0.67], "gauss_her_slope_inside_K5": bool(0.47 <= sl(rms_her_g) <= 0.67),
    "K5_iid_slope_window_F02": [-0.58, -0.38],
    "gauss_iid_slope_inside": bool(-0.58 <= sl(rms_iid_g) <= -0.38)}
out["dof_audit"] = {
    "free_parameters_declared": 0,
    "constants_that_fix_all_12_numbers": ["lambda9 (equivalently T2(C1), c2)", "Q4 (equivalently c_delta)"],
    "c_delta_equals_256Q4_over_12_identically": True,
    "already_physically_measured": "Q4 via check H (P12): c_delta = 32/9 confirmed to 4e-5",
    "residual_new_information": ["lambda9 = 8/9 (c2)", "generator vs kernel diagonal (K3)", "Schur floor (K4)"],
    "c_obs_fixed_Delta_two_species": float(C2 + C4) / 4,
    "c_obs_iid_n_to_infinity": float(C2 + C4),
    "note": "both equal Q4/12 because both are pure self-terms; the fixed-Delta statement is written "
            "in a normalisation 4x the P1-P7 c-observable, so recovers[2]/P12 constrain Q4 only."}
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "a4_lattice.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
