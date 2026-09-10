"""CE-AM2: finite energy-conserving approximate record with a coherent battery.

The state, energy scale, pulse and pointer are supplied. Exact integer block
checks and independent effect formulas do not establish physical unification.
"""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

try:
    from .ce_async_projection_memory import memory_propagate
except ImportError:
    from ce_async_projection_memory import memory_propagate


TOL = 1e-11
SIZES = (1, 2, 4, 8, 16, 32)
X = np.array([[0, 1], [1, 0]], dtype=np.int64)
I = np.eye(2, dtype=np.int64)


def lifted_record(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("support size must be a positive integer")
    length = n + 2
    dim = 4 * length
    numerator = 2 * np.eye(dim, dtype=np.int64)
    ideal_numerator = np.kron(I + X, I) + np.kron(I - X, X)
    for total in range(1, length):
        indices = [(a * length + total - a) * 2 + r for a in range(2) for r in range(2)]
        numerator[np.ix_(indices, indices)] = ideal_numerator
    energy = np.array([a + b for a in range(2) for b in range(length) for _ in range(2)], dtype=np.int64)
    # No wrap-around, floating tolerance, or finite-cutoff energy leakage here.
    assert np.array_equal(numerator.T @ numerator, 4 * np.eye(dim, dtype=np.int64))
    assert np.array_equal(numerator, numerator.T)
    assert np.all(numerator * (energy[None, :] - energy[:, None]) == 0)
    # Independent expression: resonant nearest-level exchange plus boundary projector.
    exchange = np.zeros((2*length, 2*length), dtype=np.int64)
    interior = np.zeros_like(exchange)
    for total in range(1, length):
        lo, hi = total, length+total-1
        exchange[lo, hi] = exchange[hi, lo] = 1
        interior[lo, lo] = interior[hi, hi] = 1
    assert np.array_equal(2*np.eye(dim, dtype=np.int64)-numerator,
                          np.kron(interior-exchange, I-X))
    return numerator.astype(complex) / 2, energy


def reference(n, kind):
    result = np.zeros(n + 2, dtype=complex)
    if kind == "uniform":
        result[1:n+1] = 1 / np.sqrt(n)
    elif kind == "sine":
        result[1:n+1] = np.sqrt(2/(n+1)) * np.sin(np.pi*np.arange(1, n+1)/(n+1))
    else:
        raise ValueError("unknown reference family")
    assert abs(np.vdot(result, result)-1) < TOL
    return result


def kraus(w, beta):
    length = len(beta)
    # a,n,r,b,m are output system,battery,pointer and input system,battery.
    tensor = w.reshape(2, length, 2, 2, length, 2)[:, :, :, :, :, 0]
    return np.einsum("anrbm,m->rnab", tensor, beta)


def effects(ks):
    return np.array([sum((k.conj().T @ k for k in row), np.zeros((2, 2), complex)) for row in ks])


def instrument_choi_distance(ks):
    """Trace distance of normalized CQ Choi states; not a diamond norm."""
    distance = 0.
    for r in range(2):
        vectors = [k.reshape(4, order="F") for k in ks[r]]
        actual = sum((np.outer(v, v.conj()) for v in vectors), np.zeros((4, 4), complex))
        assert np.linalg.eigvalsh(actual).min() >= -TOL
        ideal = ((I + (-1)**r * X)/2).reshape(4, order="F")
        delta = actual - np.outer(ideal, ideal.conj())
        distance += float(np.sum(np.abs(np.linalg.eigvalsh(delta)))) / 4
    return distance


def audit_size(n, kind):
    w, energy = lifted_record(n)
    length = n+2
    beta = reference(n, kind)
    ks = kraus(w, beta)
    eff = effects(ks)
    eta = float(np.vdot(beta[:-1], beta[1:]).real)
    expected_eta = (n-1)/n if kind == "uniform" else np.cos(np.pi/(n+1))
    expected = np.array([(I+eta*X)/2, (I-eta*X)/2])
    effect_error = float(np.max(np.abs(eff-expected)))
    assert abs(eta-expected_eta) < TOL and effect_error < TOL
    assert np.max(np.abs(sum(eff)-I)) < TOL
    assert min(np.linalg.eigvalsh(e).min() for e in eff) >= -TOL
    ideal_isometry = np.einsum("n,rab->rnab", beta, np.array([(I+X)/2, (I-X)/2]))
    difference_isometry = (ks-ideal_isometry).reshape(-1,2)
    assert np.max(np.abs(difference_isometry.conj().T @ difference_isometry - (1-eta)*I)) < TOL
    path = (np.eye(n, k=1) + np.eye(n, k=-1)) / 2
    best = float(np.linalg.eigvalsh(path)[-1])
    assert abs(best - np.cos(np.pi/(n+1))) < TOL
    assert eta <= best + TOL

    excited = np.zeros((2, length, 2), dtype=complex)
    excited[1, :, 0] = beta
    before = excited.ravel()
    after = w @ before
    system_e = np.repeat(np.arange(2), 2*length)
    battery_e = np.tile(np.repeat(np.arange(length), 2), 2)
    difference = np.abs(after)**2 - np.abs(before)**2
    ds = float(difference @ system_e)
    db = float(difference @ battery_e)
    assert abs(ds + .5) < TOL and abs(db - .5) < TOL
    assert abs(difference @ energy) < TOL
    mean_battery = float(np.abs(beta)**2 @ np.arange(length))
    assert abs(mean_battery - (n+1)/2) < TOL
    # G = pi(I-W)/2 gives exp(-iG)=W and commutes with bare total energy.
    g = np.pi * (np.eye(len(w))-w)/2
    assert np.max(np.abs(g * (energy[None,:]-energy[:,None]))) == 0.
    return {
        "N": n, "reference": kind, "eta": eta,
        "wrong_pointer_probability_on_X_eigenstate": (1-eta)/2,
        "normalized_CQ_Choi_trace_distance": instrument_choi_distance(ks),
        "half_diamond_distance_proved_upper_bound": float(np.sqrt(max(0.,1-eta))),
        "mean_initial_battery_energy_in_gap_units": mean_battery,
        "system_energy_change_on_excited_in_gap_units": ds,
        "battery_energy_change_on_excited_in_gap_units": db,
        "effect_formula_max_error": effect_error,
        "integer_unitarity_and_energy_commutator": True,
    }


def audit_incoherent_control():
    n = 4
    w, _ = lifted_record(n)
    for kind in ("uniform", "sine"):
        beta = reference(n, kind)
        incoherent = np.zeros((2, 2, 2), dtype=complex)
        for b in range(1, n+1):
            basis = np.eye(n+2, dtype=complex)[:, b]
            incoherent += abs(beta[b])**2 * effects(kraus(w, basis))
        assert np.max(np.abs(incoherent - np.array([I/2, I/2]))) < TOL
    return "same battery energy distribution without coherence gives effects I/2"


def audit_two_sites_and_memory():
    n = 2
    length = n+2
    local, _ = lifted_record(n)
    d = len(local)
    # Reorder (a0,n0,r0,a1,n1,r1) into (a0,a1,n0,r0,n1,r1).
    perm = np.arange(d*d).reshape(2,length,2,2,length,2).transpose(0,3,1,2,4,5).ravel()
    bare = [np.kron(local, np.eye(d)), np.kron(np.eye(d), local)]
    grouped = [m[np.ix_(perm,perm)] for m in bare]
    basis = np.array([[1,1],[-1,1]])/np.sqrt(2)  # energy basis to AM1 basis
    env = (2*length)**2
    rotation = np.kron(np.kron(basis,basis), np.eye(env))
    events = [rotation @ m @ rotation.T for m in grouped]
    beta = reference(n, "sine")
    aux = np.zeros((length,2), complex)
    aux[:,0] = beta
    system = np.array([1,0,0,1j], complex)/np.sqrt(2)
    initial = np.kron(system, np.kron(aux.ravel(),aux.ravel()))
    forward = events[1] @ (events[0] @ initial)
    reverse = events[0] @ (events[1] @ initial)
    order_error = float(np.max(np.abs(forward-reverse)))
    memory = memory_propagate(events, initial, 3*env)
    memory_error = float(np.max(np.abs(memory[-1]-forward[:3*env])))
    assert order_error < TOL and memory_error < TOL
    assert abs(np.vdot(forward,forward)-1) < TOL
    # Repeatability no-go witness: AM1 Z-basis vectors have off-diagonal H_S.
    hs_original = (I+X)/2  # energy gap units
    assert hs_original[0,1] == .5
    return {"N": n, "dimension": d*d, "order_max_error": order_error,
            "memory_max_error": memory_error,
            "repeatability_conservation_offdiagonal_obstruction_in_gap_units": .5}


def main():
    rows = [audit_size(n, kind) for n in SIZES for kind in ("uniform", "sine")]
    result = {
        "candidate": "CE-AM2", "status": "supplied finite energy-conserving approximate instrument",
        "rows": rows, "incoherent_control": audit_incoherent_control(),
        "two_sites": audit_two_sites_and_memory(), "tolerance": TOL,
        "not_derived": ["exact finite-reference sharp measurement", "actual outcome selection",
                        "autonomous pulse and reference preparation", "relativistic field locality",
                        "metric and Einstein limit", "three gauge sectors", "joint observational RMSE"],
        "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "memory_script_sha256": hashlib.sha256(Path(__file__).with_name("ce_async_projection_memory.py").read_bytes()).hexdigest()},
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
