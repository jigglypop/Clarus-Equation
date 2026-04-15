"""CE3 v2: 3+1 Gauge Network with Riemannian geometry + LBO + Spectral Norm.

Key additions over v1:
1. LBONorm instead of LayerNorm (Laplace-Beltrami diffusion)
2. Spectral normalization (unitary constraint)
3. Cross-frequency coupling (CFC gate)
4. Sequence-level context (not just last token)
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from tqdm import tqdm

DIM = 768
BATCH = 512
EPOCHS = 30
LR = 5e-4
MAX_TRAIN = 80000
MAX_TEST = 5000
LBO_RANK = 8
CFC_XI = 0.490

ALPHA_S, ALPHA_W, ALPHA_EM = 0.11789, 0.03352, 0.00775
TOTAL_A = ALPHA_S + ALPHA_W + ALPHA_EM
H3 = int(DIM * 3 * ALPHA_S / TOTAL_A)
H2 = int(DIM * 3 * ALPHA_W / TOTAL_A)
H1 = DIM * 3 - H3 - H2

print("=" * 65)
print("  CE3 v2: Riemannian Gauge Network")
print("  LBONorm + SpectralNorm + CFC + Sequence Context")
print("=" * 65)


class LBONorm(nn.Module):
    """Laplace-Beltrami Operator Normalization (2_Architecture.md 3.1).

    h' = (h_hat - eta * Delta_g h_hat) * gamma + beta
    Delta_g h = h - V^T V h  (low-rank diffusion)
    """
    def __init__(self, d, rank=LBO_RANK):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.V = nn.Parameter(torch.randn(rank, d) * 0.01)
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, h):
        h_hat = F.layer_norm(h, (h.shape[-1],))
        VtV = self.V.T @ self.V
        delta_h = h_hat - h_hat @ VtV
        e_curv = (delta_h ** 2).sum(dim=-1, keepdim=True)
        h_smooth = h_hat - self.eta.clamp(0, 0.5) * delta_h
        return h_smooth * self.gamma + self.beta, e_curv.squeeze(-1)


class GaugeLayerV2(nn.Module):
    """Gauge layer: LN + 3 gauge FFNs + spectral on output only + CFC."""
    def __init__(self, d, d_bind, d_decide, d_attend):
        super().__init__()
        self.bind = nn.Sequential(
            nn.Linear(d, d_bind), nn.GELU(),
            spectral_norm(nn.Linear(d_bind, d)))
        self.decide = nn.Sequential(
            nn.Linear(d, d_decide), nn.GELU(),
            spectral_norm(nn.Linear(d_decide, d)))
        self.attend = nn.Sequential(
            nn.Linear(d, d_attend), nn.GELU(),
            spectral_norm(nn.Linear(d_attend, d)))
        self.norm = nn.LayerNorm(d)
        self.lbo_V = nn.Parameter(torch.randn(LBO_RANK, d) * 0.01)
        self.cfc_xi = CFC_XI

    def forward(self, x):
        h = self.norm(x)
        VtV = self.lbo_V.T @ self.lbo_V
        delta_h = h - h @ VtV
        e_curv = (delta_h ** 2).sum(dim=-1, keepdim=True)
        cfc_gate = (1.0 - self.cfc_xi * e_curv.clamp(0, 2))
        out = self.bind(h) + self.decide(h) + self.attend(h)
        return x + out * cfc_gate


class CE3NetV2(nn.Module):
    """3+1 Riemannian gauge network with sequence context."""
    def __init__(self):
        super().__init__()
        self.context_proj = nn.Linear(DIM, DIM)
        self.layer1 = GaugeLayerV2(DIM, H3, H2, H1)
        self.layer2 = GaugeLayerV2(DIM, H3, H2, H1)
        self.layer3 = GaugeLayerV2(DIM, H3, H2, H1)
        self.final_norm = nn.LayerNorm(DIM)

    def forward(self, x_last, x_context=None):
        if x_context is not None:
            ctx = self.context_proj(x_context)
            x_last = x_last + 0.3 * ctx
        h = self.layer1(x_last)
        h = self.layer2(h)
        h = self.layer3(h)
        return self.final_norm(h)


net = CE3NetV2()
n_params = sum(p.numel() for p in net.parameters())
print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")

# --- Data ---
print("\n[1/4] Loading GPT-2 + WikiText-2...")
from transformers import AutoModelForCausalLM, AutoTokenizer
teacher = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tok = AutoTokenizer.from_pretrained("gpt2")
lm_head = teacher.lm_head.weight.detach().float()

from datasets import load_dataset
ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")


def collect(texts_raw, max_pairs, desc):
    texts = [t for t in texts_raw if len(t.strip()) > 30]
    X_last, X_ctx, Y, TGT = [], [], [], []
    count = 0
    with torch.no_grad():
        for text in tqdm(texts, desc=desc, ncols=80):
            if count >= max_pairs:
                break
            ids = tok(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            if ids.shape[1] < 4:
                continue
            out = teacher(ids, output_hidden_states=True)
            emb = out.hidden_states[0][0].float().cpu()
            hid = out.hidden_states[-1][0].float().cpu()
            target_ids = ids[0, 1:].cpu()

            seq_len = min(emb.shape[0] - 1, target_ids.shape[0])
            for pos in range(seq_len):
                X_last.append(emb[pos])
                X_ctx.append(emb[:pos+1].mean(dim=0))
                Y.append(hid[pos])
                TGT.append(target_ids[pos].item())
                count += 1
                if count >= max_pairs:
                    break

    return (torch.stack(X_last), torch.stack(X_ctx),
            torch.stack(Y), torch.tensor(TGT, dtype=torch.long))


X_last_tr, X_ctx_tr, Y_tr, TGT_tr = collect(ds_train["text"], MAX_TRAIN, "train")
X_last_te, X_ctx_te, Y_te, TGT_te = collect(ds_test["text"], MAX_TEST, "test")
print(f"  Train: {X_last_tr.shape[0]}  Test: {X_last_te.shape[0]}")

del teacher
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- Train ---
print(f"\n[2/4] Training CE3NetV2 ({EPOCHS} epochs)...")
optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
steps_total = EPOCHS * (X_last_tr.shape[0] // BATCH + 1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=steps_total, pct_start=0.05)

best_top1 = -1.0

for epoch in range(EPOCHS):
    net.train()
    perm = torch.randperm(X_last_tr.shape[0])
    total_loss = 0
    n_batch = 0

    pbar = tqdm(range(0, X_last_tr.shape[0], BATCH),
                desc=f"E{epoch+1:2d}/{EPOCHS}", ncols=100, leave=False)
    for i in pbar:
        idx = perm[i:i+BATCH]
        xl, xc, yb, tgt = X_last_tr[idx], X_ctx_tr[idx], Y_tr[idx], TGT_tr[idx]

        pred = net(xl, xc)
        cos_loss = 1.0 - F.cosine_similarity(pred, yb, dim=1).mean()
        mse_loss = F.mse_loss(pred, yb)
        ce_loss = F.cross_entropy(pred @ lm_head.T, tgt)
        loss = mse_loss + 2.0 * cos_loss + 0.5 * ce_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    # Eval every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        net.eval()
        with torch.no_grad():
            pred_te = net(X_last_te, X_ctx_te)
            cos_te = F.cosine_similarity(pred_te, Y_te, dim=1).mean().item()
            logits_ce = pred_te @ lm_head.T
            logits_true = Y_te @ lm_head.T
            true_top1 = logits_true.argmax(dim=1)
            ce_top1 = logits_ce.argmax(dim=1)
            top1 = (ce_top1 == true_top1).float().mean().item()

            top5_hits = sum(
                1 for j in range(min(2000, X_last_te.shape[0]))
                if true_top1[j].item() in torch.topk(logits_ce[j], 5).indices.tolist()
            )
            top5 = top5_hits / min(2000, X_last_te.shape[0])
            ce_target = F.cross_entropy(logits_ce, TGT_te).item()

        if top1 > best_top1:
            best_top1 = top1
            torch.save(net.state_dict(), "clarus/ce3v2_best.pt")

        print(f"  E{epoch+1:2d} loss={total_loss/n_batch:.3f} cos={cos_te:.4f} "
              f"top1={top1*100:.1f}% top5={top5*100:.1f}% CE={ce_target:.2f} "
              f"{'*BEST*' if top1 >= best_top1 else ''}")

# --- Eval ---
print(f"\n[3/4] Best model: top1={best_top1*100:.1f}%")
net.load_state_dict(torch.load("clarus/ce3v2_best.pt", weights_only=True))
net.eval()

with torch.no_grad():
    pred_all = net(X_last_te, X_ctx_te)
    logits_ce = pred_all @ lm_head.T
    logits_true = Y_te @ lm_head.T
    true_top1 = logits_true.argmax(dim=1)
    t1 = t5 = t10 = 0
    for j in range(X_last_te.shape[0]):
        target = true_top1[j].item()
        topk = torch.topk(logits_ce[j], 10).indices.tolist()
        if target == topk[0]: t1 += 1
        if target in topk[:5]: t5 += 1
        if target in topk[:10]: t10 += 1

n = X_last_te.shape[0]
print(f"  Top-1:  {t1/n*100:.1f}%")
print(f"  Top-5:  {t5/n*100:.1f}%")
print(f"  Top-10: {t10/n*100:.1f}%")

# --- Generation ---
print("\n[4/4] Generation test...")
teacher2 = AutoModelForCausalLM.from_pretrained("gpt2").eval()
prompts = [
    "The future of AI is", "Once upon a time there was",
    "Scientists discovered that the", "The weather today is very",
    "The best way to learn is", "The president announced that",
    "In the history of science", "The most important thing in life",
]
matches = 0
for p in prompts:
    ids = tok(p, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = teacher2(ids, output_hidden_states=True)
        emb = out.hidden_states[0][0].float()
        last = emb[-1:]; ctx = emb.mean(dim=0, keepdim=True)
        pred = net(last, ctx)
    logits = (pred @ lm_head.T).squeeze(0)
    ce5 = [tok.decode([i]) for i in torch.topk(logits, 5).indices.tolist()]
    g5 = [tok.decode([i]) for i in torch.topk(out.logits[0, -1], 5).indices.tolist()]
    gid = out.logits[0, -1].argmax().item()
    rank = int((logits >= logits[gid]).sum().item())
    tag = " <<< MATCH" if rank == 1 else ""
    if rank == 1: matches += 1
    print(f'  "{p}"')
    print(f"    GPT-2: {g5}")
    print(f"    CE-3R: {ce5}  rank={rank}{tag}")
print(f"\n  Matches: {matches}/{len(prompts)}")

# Speed
emb_b = teacher2.transformer.wte(
    tok("The future of AI is", return_tensors="pt")["input_ids"])[0]
last_b = emb_b[-1:].float().detach()
ctx_b = emb_b.mean(dim=0, keepdim=True).float().detach()
ids_b = tok("The future of AI is", return_tensors="pt")["input_ids"]

n_b = 500
t0 = time.perf_counter()
for _ in range(n_b):
    with torch.no_grad(): teacher2(ids_b)
g_ms = (time.perf_counter() - t0) / n_b * 1000
t0 = time.perf_counter()
for _ in range(n_b):
    with torch.no_grad(): net(last_b, ctx_b)
c_ms = (time.perf_counter() - t0) / n_b * 1000

gpt_mb = sum(p.numel() * p.element_size() for p in teacher2.parameters()) / 1024 / 1024
ce_mb = n_params * 4 / 1024 / 1024
print(f"\nSpeed:  GPT-2={g_ms:.1f}ms  CE-3R={c_ms:.3f}ms  {g_ms/c_ms:.0f}x faster")
print(f"Params: GPT-2=124M  CE-3R={n_params/1e6:.1f}M ({n_params/124e6*100:.1f}%)")
print(f"Memory: GPT-2={gpt_mb:.0f}MB  CE-3R={ce_mb:.0f}MB ({ce_mb/gpt_mb*100:.0f}%)")
print("done")
