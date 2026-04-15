"""Train CE 3+1 Gauge Network: distill GPT-2 (12L) into 3 gauge layers.

d=3 -> SU(3):SU(2):U(1) = 74:21:5 channel split per layer.
Each layer: Bind(74%) + Decide(21%) + Attend(5%) + Smooth(LN)
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

print("=" * 65)
print("  CE 3+1 Gauge Network Training")
print("  d(d-3)=0 => d=3 => 3 layers")
print("  SU(3):SU(2):U(1) = 74.1% : 21.1% : 4.9%")
print("=" * 65)

# --- Config ---
DIM = 768
BATCH = 512
EPOCHS = 50
LR = 5e-4
MAX_TRAIN = 80000
MAX_TEST = 5000

# Gauge ratios from CE coupling constants
ALPHA_S, ALPHA_W, ALPHA_EM = 0.11789, 0.03352, 0.00775
TOTAL_A = ALPHA_S + ALPHA_W + ALPHA_EM
H_BIND = int(DIM * 3 * ALPHA_S / TOTAL_A)
H_DECIDE = int(DIM * 3 * ALPHA_W / TOTAL_A)
H_ATTEND = DIM * 3 - H_BIND - H_DECIDE

print(f"\n  Hidden widths: Bind={H_BIND} Decide={H_DECIDE} Attend={H_ATTEND}")
print(f"  Ratios: {H_BIND/(H_BIND+H_DECIDE+H_ATTEND)*100:.1f}% / "
      f"{H_DECIDE/(H_BIND+H_DECIDE+H_ATTEND)*100:.1f}% / "
      f"{H_ATTEND/(H_BIND+H_DECIDE+H_ATTEND)*100:.1f}%")

# --- Model ---
class GaugeLayer(nn.Module):
    """One d=3 gauge layer: 3 parallel FFNs (SU3+SU2+U1) + residual + LN."""
    def __init__(self, d, d_bind, d_decide, d_attend):
        super().__init__()
        self.bind = nn.Sequential(
            nn.Linear(d, d_bind), nn.GELU(), nn.Linear(d_bind, d))
        self.decide = nn.Sequential(
            nn.Linear(d, d_decide), nn.GELU(), nn.Linear(d_decide, d))
        self.attend = nn.Sequential(
            nn.Linear(d, d_attend), nn.GELU(), nn.Linear(d_attend, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.bind(x) + self.decide(x) + self.attend(x))


class CE3GaugeNet(nn.Module):
    """3+1 gauge network: 3 GaugeLayers + final LN (the +1 Smooth)."""
    def __init__(self):
        super().__init__()
        self.layer1 = GaugeLayer(DIM, H_BIND, H_DECIDE, H_ATTEND)
        self.layer2 = GaugeLayer(DIM, H_BIND, H_DECIDE, H_ATTEND)
        self.layer3 = GaugeLayer(DIM, H_BIND, H_DECIDE, H_ATTEND)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


# --- Data collection ---
print("\n[1/4] Loading GPT-2 teacher...")
from transformers import AutoModelForCausalLM, AutoTokenizer
teacher = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_head = teacher.lm_head.weight.detach().float()
ln_f = teacher.transformer.ln_f

print("[2/4] Collecting training pairs from WikiText-2...")
from datasets import load_dataset
ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
train_texts = [t for t in ds_train["text"] if len(t.strip()) > 30]
test_texts = [t for t in ds_test["text"] if len(t.strip()) > 30]


def collect(texts, max_pairs, desc="collect"):
    X, Y, TGT = [], [], []
    with torch.no_grad():
        for text in tqdm(texts, desc=desc, ncols=80):
            if sum(x.shape[0] for x in X) >= max_pairs:
                break
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            if ids.shape[1] < 4:
                continue
            out = teacher(ids, output_hidden_states=True)
            emb = out.hidden_states[0][0].float().cpu()
            hid = out.hidden_states[-1][0].float().cpu()
            target_ids = ids[0, 1:].cpu()
            X.append(emb[:-1])
            Y.append(hid[:-1])
            TGT.append(target_ids)
    X = torch.cat(X)[:max_pairs]
    Y = torch.cat(Y)[:max_pairs]
    TGT = torch.cat(TGT)[:max_pairs]
    return X, Y, TGT


X_train, Y_train, TGT_train = collect(train_texts, MAX_TRAIN, "train")
X_test, Y_test, TGT_test = collect(test_texts, MAX_TEST, "test")
print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

del teacher
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- Training ---
print("\n[3/4] Training CE3GaugeNet...")
net = CE3GaugeNet()
n_params = sum(p.numel() for p in net.parameters())
print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
print(f"  vs GPT-2:   124M ({n_params/124e6*100:.1f}%)")

optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=EPOCHS * (X_train.shape[0] // BATCH + 1),
    pct_start=0.1, anneal_strategy="cos",
)

best_top1 = 0.0
log = []

for epoch in range(EPOCHS):
    net.train()
    perm = torch.randperm(X_train.shape[0])
    epoch_loss = 0.0
    n_batch = 0

    pbar = tqdm(range(0, X_train.shape[0], BATCH),
                desc=f"Epoch {epoch+1:2d}/{EPOCHS}", ncols=100, leave=False)
    for i in pbar:
        idx = perm[i:i+BATCH]
        xb, yb = X_train[idx], Y_train[idx]
        tgt = TGT_train[idx]

        pred = net(xb)

        # Loss 1: match hidden state (cosine + MSE)
        cos_loss = 1.0 - F.cosine_similarity(pred, yb, dim=1).mean()
        mse_loss = F.mse_loss(pred, yb)

        # Loss 2: match next-token prediction (cross-entropy distillation)
        pred_logits = pred @ lm_head.T
        ce_loss = F.cross_entropy(pred_logits, tgt)

        loss = mse_loss + 2.0 * cos_loss + 0.5 * ce_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    # Eval
    net.eval()
    with torch.no_grad():
        pred_test = net(X_test)
        cos_test = F.cosine_similarity(pred_test, Y_test, dim=1).mean().item()

        logits_ce = pred_test @ lm_head.T
        logits_true = Y_test @ lm_head.T
        true_top1 = logits_true.argmax(dim=1)
        ce_top1 = logits_ce.argmax(dim=1)
        top1_acc = (ce_top1 == true_top1).float().mean().item()

        # Top-5
        top5_hits = 0
        for j in range(min(X_test.shape[0], 2000)):
            target = true_top1[j].item()
            ce_top5 = torch.topk(logits_ce[j], 5).indices.tolist()
            if target in ce_top5:
                top5_hits += 1
        top5_acc = top5_hits / min(X_test.shape[0], 2000)

        # CE on actual target tokens
        ce_target = F.cross_entropy(logits_ce, TGT_test).item()

    log.append({
        "epoch": epoch + 1,
        "loss": epoch_loss / n_batch,
        "cos": cos_test,
        "top1": top1_acc,
        "top5": top5_acc,
        "ce_target": ce_target,
    })

    if top1_acc > best_top1:
        best_top1 = top1_acc
        torch.save(net.state_dict(), "clarus/ce3gauge_best.pt")

    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        print(f"  E{epoch+1:2d} loss={epoch_loss/n_batch:.4f} "
              f"cos={cos_test:.4f} top1={top1_acc*100:.1f}% "
              f"top5={top5_acc*100:.1f}% CE={ce_target:.2f} "
              f"{'*BEST*' if top1_acc >= best_top1 else ''}")

# --- Evaluation ---
print(f"\n[4/4] Final evaluation (best top1={best_top1*100:.1f}%)...")
net.load_state_dict(torch.load("clarus/ce3gauge_best.pt", weights_only=True))
net.eval()

# Full test accuracy
with torch.no_grad():
    pred_all = net(X_test)
    logits_ce = pred_all @ lm_head.T
    logits_true = Y_test @ lm_head.T
    true_top1 = logits_true.argmax(dim=1)

    t1 = t5 = t10 = 0
    for j in range(X_test.shape[0]):
        target = true_top1[j].item()
        topk = torch.topk(logits_ce[j], 10).indices.tolist()
        if target == topk[0]: t1 += 1
        if target in topk[:5]: t5 += 1
        if target in topk[:10]: t10 += 1

n = X_test.shape[0]
print(f"  Top-1:  {t1/n*100:.1f}%")
print(f"  Top-5:  {t5/n*100:.1f}%")
print(f"  Top-10: {t10/n*100:.1f}%")

# Generation
print("\n" + "=" * 65)
print("  Generation: CE 3-Gauge vs GPT-2")
print("=" * 65)

teacher2 = AutoModelForCausalLM.from_pretrained("gpt2").eval()
for p in ["The future of AI is", "Once upon a time there was",
          "Scientists discovered that the", "The weather today is very",
          "The best way to learn is", "The president announced that",
          "In the history of science", "The most important thing in life"]:
    ids = tokenizer(p, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = teacher2(ids, output_hidden_states=True)
        emb = out.hidden_states[0][0, -1:].float()
        pred = net(emb)
    logits = (pred @ lm_head.T).squeeze(0)
    ce5 = [tokenizer.decode([i]) for i in torch.topk(logits, 5).indices.tolist()]
    g5 = [tokenizer.decode([i]) for i in torch.topk(out.logits[0, -1], 5).indices.tolist()]
    gid = out.logits[0, -1].argmax().item()
    rank = int((logits >= logits[gid]).sum().item())
    tag = " <<< MATCH" if rank == 1 else ""
    print(f'  "{p}"')
    print(f"    GPT-2: {g5}")
    print(f"    CE-3G: {ce5}  rank={rank}{tag}")

# Speed
emb_b = teacher2.transformer.wte(
    tokenizer("The future of AI is", return_tensors="pt")["input_ids"]
)[0, -1:].float().detach()
ids_b = tokenizer("The future of AI is", return_tensors="pt")["input_ids"]

n_b = 500
t0 = time.perf_counter()
for _ in range(n_b):
    with torch.no_grad():
        teacher2(ids_b)
g_ms = (time.perf_counter() - t0) / n_b * 1000

t0 = time.perf_counter()
for _ in range(n_b):
    with torch.no_grad():
        net(emb_b)
c_ms = (time.perf_counter() - t0) / n_b * 1000

gpt_mb = sum(p.numel() * p.element_size() for p in teacher2.parameters()) / 1024 / 1024
ce_mb = n_params * 4 / 1024 / 1024

print(f"\nSpeed:  GPT-2={g_ms:.1f}ms  CE-3G={c_ms:.3f}ms  {g_ms/c_ms:.0f}x faster")
print(f"Params: GPT-2=124M  CE-3G={n_params/1e6:.1f}M ({n_params/124e6*100:.1f}%)")
print(f"Memory: GPT-2={gpt_mb:.0f}MB  CE-3G={ce_mb:.0f}MB ({ce_mb/gpt_mb*100:.0f}%)")
print("done")
