"""V14 route-explorer toy bench: structurally distinct latch/binding families.

Read-only w.r.t. the repo: imports the frozen v13 episode generator and the
frozen 20-channel raw-sequence encoding; all candidate models live here.

Protocol (fixed for every candidate, matching examples/agi/local_cloud_v13_run.py):
  train pool = balanced split, 24 cells, T=4, sigma=0.04, 192 episodes
  Adam lr=0.01, weight_decay=1e-4, grad clip=1.0, 200 epochs, full batch,
  BCEWithLogits; evaluation on the five v13 panels, 256 episodes each.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "reality_stone" / "python"))

from reality_stone.clarus.local_cloud_benchmark import LocalCloudBenchmarkConfig  # noqa: E402
from reality_stone.clarus.local_cloud_v13_benchmark import (  # noqa: E402
    V13_PANELS, generate_episodes_v2, panel_configs_v13,
)

IN = 20


def raw_sequences(episodes) -> np.ndarray:
    rows = []
    for ep in episodes:
        rows.append(tuple(
            tuple(v for row in ob.local for v in row) + ob.shared for ob in ep.observations
        ))
    out = np.asarray(rows, dtype=np.float32)
    if out.ndim != 3 or out.shape[2] != IN or not np.all(np.isfinite(out)):
        raise ValueError("raw input must be finite (episodes, ticks, 20)")
    return out


class LatchBilinear(nn.Module):
    """(A) two salience-latched slots + explicit bilinear (outer-product) readout.
    s' = (1-g) s + g tanh(U x);  yhat = <w, s_c s_b^T> + b."""

    def __init__(self, db: int = 8, dc: int = 4, linear_readout: bool = False,
                 hard_gate: bool = False) -> None:
        super().__init__()
        self.db, self.dc = db, dc
        self.linear_readout, self.hard_gate = linear_readout, hard_gate
        self.Ub = nn.Linear(IN, db, bias=False)
        self.Uc = nn.Linear(IN, dc, bias=False)
        self.gb, self.gc = nn.Linear(IN, 1), nn.Linear(IN, 1)
        nn.init.constant_(self.gb.bias, -1.0)
        nn.init.constant_(self.gc.bias, -1.0)
        width = (db + dc) if linear_readout else (db * dc)
        self.out = nn.Linear(width, 1)

    def gates(self, x):
        if self.hard_gate:
            # (G) oracle salience gate: no learned parameters, thresholded
            # channel-group energy.  Target-aware (uses the known 16 local /
            # 4 shared channel layout); a diagnostic upper bound, not a route.
            local = (x[:, :16] * x[:, :16]).sum(-1, keepdim=True)
            shared = (x[:, 16:] * x[:, 16:]).sum(-1, keepdim=True)
            return (local > 0.5).float(), (shared > 0.5).float()
        return torch.sigmoid(self.gb(x)), torch.sigmoid(self.gc(x))

    def forward(self, seq):
        n = seq.shape[0]
        sb = seq.new_zeros(n, self.db)
        sc = seq.new_zeros(n, self.dc)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            gb, gc = self.gates(x)
            sb = (1 - gb) * sb + gb * torch.tanh(self.Ub(x))
            sc = (1 - gc) * sc + gc * torch.tanh(self.Uc(x))
        if self.linear_readout:
            return self.out(torch.cat([sb, sc], -1)).squeeze(-1)
        outer = (sc.unsqueeze(2) * sb.unsqueeze(1)).reshape(n, -1)
        return self.out(outer).squeeze(-1)


class HRRBinding(nn.Module):
    """(B) multiplicative recurrent binding by circular convolution (HRR/TPR family).
    h' = (1-g) h + g (circ(h, rho(x)) + tanh(U x));  yhat = <w, h>.
    A linear readout on a circular-convolution-bound state is already a
    bilinear form in the two encoded latents."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.d = d
        self.U = nn.Linear(IN, d, bias=False)
        self.V = nn.Linear(IN, d, bias=False)
        self.g = nn.Linear(IN, 1)
        nn.init.constant_(self.g.bias, -1.0)
        self.out = nn.Linear(d, 1)
        idx = torch.arange(d)
        self.register_buffer("shift", (idx[:, None] - idx[None, :]) % d)
        delta = torch.zeros(d)
        delta[0] = 1.0
        self.register_buffer("delta", delta)

    def circ(self, a, b):
        return torch.einsum("ni,nki->nk", a, b[:, self.shift])

    def forward(self, seq):
        h = seq.new_zeros(seq.shape[0], self.d)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            g = torch.sigmoid(self.g(x))
            rho = torch.tanh(self.V(x)) + self.delta
            h = (1 - g) * h + g * (self.circ(h, rho) + torch.tanh(self.U(x)))
        return self.out(h).squeeze(-1)


class OuterMemory(nn.Module):
    """(C)/(D) staged latch + gated outer-product write into a matrix memory.
    s' = (1-gs) s + gs tanh(Ws x);  M' = M + gm key(x) s^T;  yhat = <W, M>.
    key = softmax(Wk x): key-value slot routing (simplex key), route C.
    key = Wk x: fast-weight outer-product memory (free key), route D."""

    def __init__(self, k: int = 4, d: int = 4, softmax_key: bool = True) -> None:
        super().__init__()
        self.k, self.d, self.softmax_key = k, d, softmax_key
        self.Ws = nn.Linear(IN, d, bias=False)
        self.Wk = nn.Linear(IN, k, bias=False)
        self.gs, self.gm = nn.Linear(IN, 1), nn.Linear(IN, 1)
        nn.init.constant_(self.gs.bias, -1.0)
        nn.init.constant_(self.gm.bias, -1.0)
        self.out = nn.Linear(k * d, 1)

    def forward(self, seq):
        n = seq.shape[0]
        s = seq.new_zeros(n, self.d)
        memory = seq.new_zeros(n, self.k, self.d)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            gs, gm = torch.sigmoid(self.gs(x)), torch.sigmoid(self.gm(x))
            key = self.Wk(x)
            if self.softmax_key:
                key = torch.softmax(key, -1)
            memory = memory + gm.unsqueeze(-1) * key.unsqueeze(2) * s.unsqueeze(1)
            s = (1 - gs) * s + gs * torch.tanh(self.Ws(x))
        return self.out(memory.reshape(n, -1)).squeeze(-1)


class RotationQuadratic(nn.Module):
    """(E) structurally orthogonal recurrence (block 2x2 rotations, all
    eigenvalues of unit modulus by construction, no write gate) plus a
    quadratic readout.  h' = R h + B x;  yhat = <w,h> + h^T Q h."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        if d % 2:
            raise ValueError("d must be even")
        self.d = d
        self.theta = nn.Parameter(torch.zeros(d // 2))
        self.B = nn.Linear(IN, d, bias=False)
        self.lin = nn.Linear(d, 1)
        self.P = nn.Parameter(0.1 * torch.randn(d, d))

    def rotate(self, h):
        cos, sin = torch.cos(self.theta), torch.sin(self.theta)
        a, b = h[:, 0::2], h[:, 1::2]
        return torch.stack([cos * a - sin * b, sin * a + cos * b], -1).reshape(h.shape)

    def forward(self, seq):
        h = seq.new_zeros(seq.shape[0], self.d)
        for t in range(seq.shape[1]):
            h = self.rotate(h) + self.B(seq[:, t])
        quad = self.P + self.P.T
        return self.lin(h).squeeze(-1) + torch.einsum("ni,ij,nj->n", h, quad, h)


class GatedIntegratorQuadratic(nn.Module):
    """(F) single gated additive integrator, no slot separation, quadratic
    readout.  h' = h + g tanh(B x);  yhat = <w,h> + h^T Q h."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.d = d
        self.B = nn.Linear(IN, d, bias=False)
        self.g = nn.Linear(IN, 1)
        nn.init.constant_(self.g.bias, -1.0)
        self.lin = nn.Linear(d, 1)
        self.P = nn.Parameter(0.1 * torch.randn(d, d))

    def forward(self, seq):
        h = seq.new_zeros(seq.shape[0], self.d)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            h = h + torch.sigmoid(self.g(x)) * torch.tanh(self.B(x))
        quad = self.P + self.P.T
        return self.lin(h).squeeze(-1) + torch.einsum("ni,ij,nj->n", h, quad, h)


class GRU20(nn.Module):
    """(R) reference: the frozen gru20 baseline architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(IN, 20, batch_first=True)
        self.out = nn.Linear(20, 1)

    def forward(self, seq):
        states, _ = self.rnn(seq)
        return self.out(states[:, -1]).squeeze(-1)


ROUTES = {
    "A_latch_bilinear": lambda: LatchBilinear(8, 4),
    "B_hrr_binding": lambda: HRRBinding(8),
    "C_kv_slot_softmax": lambda: OuterMemory(4, 4, softmax_key=True),
    "D_fastweight_outer": lambda: OuterMemory(4, 4, softmax_key=False),
    "E_rotation_quadratic": lambda: RotationQuadratic(8),
    "F_integrator_quadratic": lambda: GatedIntegratorQuadratic(8),
    "G_hardlatch_bilinear": lambda: LatchBilinear(8, 4, hard_gate=True),
    "H_latch_linear_ctrl": lambda: LatchBilinear(8, 4, linear_readout=True),
    "R_gru20": GRU20,
}


class DelayLineBilinear(nn.Module):
    """(I) tapped delay line: strictly nilpotent shift transition (no gate, no
    marginal stability), state = last k encoded ticks, bilinear readout on the
    outer product of two taps.  Retention by translation, not by eigenvalue 1;
    predicts exact collapse as soon as T exceeds k."""

    def __init__(self, k: int = 4, d: int = 5) -> None:
        super().__init__()
        self.k, self.d = k, d
        self.enc = nn.Linear(IN, d, bias=False)
        self.out = nn.Linear(d * d, 1)

    def forward(self, seq):
        n = seq.shape[0]
        taps = [seq.new_zeros(n, self.d) for _ in range(self.k)]
        for t in range(seq.shape[1]):
            taps = [torch.tanh(self.enc(seq[:, t]))] + taps[:-1]
        outer = (taps[-1].unsqueeze(2) * taps[-2].unsqueeze(1)).reshape(n, -1)
        return self.out(outer).squeeze(-1)


class FactoredMultiplicativeRNN(nn.Module):
    """(J) gated multiplicative (sigma-pi / mRNN) transition: the product is
    formed inside the transition in a learned diagonal factor basis rather
    than by circular convolution (B) or by an outer product (C/D).
    h' = (1-g) h + g tanh(U x + Wh (rho(x) * (V h)))."""

    def __init__(self, d: int = 8, f: int = 8) -> None:
        super().__init__()
        self.d = d
        self.U = nn.Linear(IN, d, bias=False)
        self.Rho = nn.Linear(IN, f, bias=False)
        self.V = nn.Linear(d, f, bias=False)
        self.Wh = nn.Linear(f, d, bias=False)
        self.g = nn.Linear(IN, 1)
        nn.init.constant_(self.g.bias, -1.0)
        self.out = nn.Linear(d, 1)

    def forward(self, seq):
        h = seq.new_zeros(seq.shape[0], self.d)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            g = torch.sigmoid(self.g(x))
            candidate = torch.tanh(self.U(x) + self.Wh(torch.tanh(self.Rho(x)) * self.V(h)))
            h = (1 - g) * h + g * candidate
        return self.out(h).squeeze(-1)


ROUTES["I_delayline_bilinear"] = lambda: DelayLineBilinear(4, 5)
ROUTES["J_factored_mrnn"] = lambda: FactoredMultiplicativeRNN(8, 8)


class EnergyGate(nn.Module):
    """Learned salience gate that is an EVEN function of the drive:
    g = sigmoid(a * ||m * x||^2 + b).  Motivated by the diagnosis that a
    sigmoid-of-linear gate cannot detect a sign-varying signal tick, because
    salience (input energy / surprise) is quadratic, not linear, in x."""

    def __init__(self, width: int = IN) -> None:
        super().__init__()
        self.mask = nn.Parameter(torch.ones(width) * 0.3)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        energy = ((self.mask * x) ** 2).sum(-1, keepdim=True)
        return torch.sigmoid(self.scale * energy + self.bias)


class EnergyLatchBilinear(nn.Module):
    """(K) two salience latches with learned ENERGY gates + bilinear readout.
    s' = (1-g) s + g tanh(U x), g = sigmoid(a ||m * x||^2 + b);
    yhat = <W, s_c s_b^T>.  Marginal stability structural (convex, r = 1)."""

    def __init__(self, db: int = 8, dc: int = 4) -> None:
        super().__init__()
        self.db, self.dc = db, dc
        self.Ub = nn.Linear(IN, db, bias=False)
        self.Uc = nn.Linear(IN, dc, bias=False)
        self.gb, self.gc = EnergyGate(), EnergyGate()
        self.out = nn.Linear(db * dc, 1)

    def forward(self, seq):
        n = seq.shape[0]
        sb = seq.new_zeros(n, self.db)
        sc = seq.new_zeros(n, self.dc)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            gb, gc = self.gb(x), self.gc(x)
            sb = (1 - gb) * sb + gb * torch.tanh(self.Ub(x))
            sc = (1 - gc) * sc + gc * torch.tanh(self.Uc(x))
        outer = (sc.unsqueeze(2) * sb.unsqueeze(1)).reshape(n, -1)
        return self.out(outer).squeeze(-1)


class EnergyHRRBinding(HRRBinding):
    """(L) route B with the energy gate substituted for the linear gate:
    tests whether the gate fix transfers across the binding family."""

    def __init__(self, d: int = 8) -> None:
        super().__init__(d)
        self.energy = EnergyGate()
        self.g = None

    def forward(self, seq):
        h = seq.new_zeros(seq.shape[0], self.d)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            g = self.energy(x)
            rho = torch.tanh(self.V(x)) + self.delta
            h = (1 - g) * h + g * (self.circ(h, rho) + torch.tanh(self.U(x)))
        return self.out(h).squeeze(-1)


class EnergyOuterMemory(OuterMemory):
    """(M) route C/D with energy gates on both the staging latch and the
    outer-product write."""

    def __init__(self, k: int = 4, d: int = 4, softmax_key: bool = True) -> None:
        super().__init__(k, d, softmax_key)
        self.egs, self.egm = EnergyGate(), EnergyGate()
        self.gs = None
        self.gm = None

    def forward(self, seq):
        n = seq.shape[0]
        s = seq.new_zeros(n, self.d)
        memory = seq.new_zeros(n, self.k, self.d)
        for t in range(seq.shape[1]):
            x = seq[:, t]
            gs, gm = self.egs(x), self.egm(x)
            key = self.Wk(x)
            if self.softmax_key:
                key = torch.softmax(key, -1)
            memory = memory + gm.unsqueeze(-1) * key.unsqueeze(2) * s.unsqueeze(1)
            s = (1 - gs) * s + gs * torch.tanh(self.Ws(x))
        return self.out(memory.reshape(n, -1)).squeeze(-1)


ROUTES["K_energylatch_bilinear"] = lambda: EnergyLatchBilinear(8, 4)
ROUTES["L_energy_hrr"] = lambda: EnergyHRRBinding(8)
ROUTES["M_energy_kv_slot"] = lambda: EnergyOuterMemory(4, 4, softmax_key=True)


class AsymEnergyLatchBilinear(EnergyLatchBilinear):
    """(N) route K with the two salience gates initialised asymmetrically, so
    the bits latch and the context latch are not forced to open on the same
    ticks.  Isolates C1(ii) (slot separation) from readout capacity."""

    def __init__(self, db: int = 8, dc: int = 4, seed_offset: int = 0) -> None:
        super().__init__(db, dc)
        with torch.no_grad():
            self.gb.mask.copy_(0.3 * torch.randn(IN).abs() + 0.05)
            self.gc.mask.copy_(0.3 * torch.randn(IN).abs() + 0.05)


ROUTES["N_asym_energylatch_bilinear"] = lambda: AsymEnergyLatchBilinear(8, 4)


def train_one(name, train_x, train_y, seed, epochs):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = ROUTES[name]()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xs = torch.from_numpy(train_x)
    ys = torch.as_tensor((np.asarray(train_y, dtype=np.float32) + 1.0) / 2.0)
    model.train()
    loss_value = float("nan")
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xs), ys)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss for " + name)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_value = float(loss.item())
    return model, loss_value


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x)).numpy()
    if not np.all(np.isfinite(scores)):
        return float("nan")
    return float(np.mean(np.where(scores >= 0.0, 1, -1) == np.asarray(y)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="9000,9001,9002,9003,9004,9005")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--train-episodes", type=int, default=192)
    ap.add_argument("--evaluation-episodes", type=int, default=256)
    ap.add_argument("--routes", type=str, default=",".join(ROUTES))
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    names = list(args.routes.split(","))

    records = []
    for seed in seeds:
        cfg = LocalCloudBenchmarkConfig(
            train_episodes=args.train_episodes,
            evaluation_episodes=args.evaluation_episodes,
            episode_steps=4, noise_sigma=0.04,
        )
        tr = generate_episodes_v2(seed, args.train_episodes, cfg, split="train",
                                  condition_split="balanced")
        train_x = raw_sequences(tr)
        train_y = tuple(e.target for e in tr)
        panels = panel_configs_v13(train_episodes=args.train_episodes,
                                   evaluation_episodes=args.evaluation_episodes,
                                   heldout_split="balanced")
        evalsets = {}
        for panel in V13_PANELS:
            pconf, csplit = panels[panel]
            ev = generate_episodes_v2(seed, args.evaluation_episodes, pconf,
                                      split="evaluation", condition_split=csplit)
            evalsets[panel] = (raw_sequences(ev), tuple(e.target for e in ev))
        for name in names:
            model, loss = train_one(name, train_x, train_y, seed * 100 + 50, args.epochs)
            rec = {"route": name, "seed": seed, "train_loss": loss,
                   "parameters": int(sum(p.numel() for p in model.parameters())),
                   "train_accuracy": accuracy(model, train_x, train_y)}
            for panel, (px, py) in evalsets.items():
                rec[panel] = accuracy(model, px, py)
            records.append(rec)
            print(json.dumps(rec), flush=True)

    summary = {}
    for name in names:
        rs = [r for r in records if r["route"] == name]
        summary[name] = {
            "parameters": rs[0]["parameters"],
            "train": float(np.mean([r["train_accuracy"] for r in rs])),
        }
        for panel in V13_PANELS:
            values = [r[panel] for r in rs]
            summary[name][panel] = {"mean": float(np.mean(values)),
                                    "min": float(np.min(values)),
                                    "max": float(np.max(values))}
    out = {"protocol": {"seeds": seeds, "epochs": args.epochs,
                        "train_episodes": args.train_episodes,
                        "evaluation_episodes": args.evaluation_episodes,
                        "optimizer": "Adam lr=0.01 wd=1e-4 clip=1.0 full-batch BCEWithLogits",
                        "split": "balanced", "torch": torch.__version__},
           "records": records, "summary": summary}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
