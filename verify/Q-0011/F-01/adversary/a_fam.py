"""Tree families for the K6 sandwich hunt (adversary-built, beyond the card's battery)."""
from __future__ import annotations

import math

import numpy as np


def chain(n):
    return [-1] + list(range(n - 1))


def star(n):
    return [-1] + [0] * (n - 1)


def kary(k, depth):
    parent = [-1]
    frontier = [0]
    for _ in range(depth):
        nxt = []
        for f in frontier:
            for _ in range(k):
                parent.append(f)
                nxt.append(len(parent) - 1)
        frontier = nxt
    return parent


def split_stars(n, fracs):
    """root + k branches with prescribed size fractions; each branch = branch-root + leaves."""
    parent = [-1]
    tot = sum(fracs)
    sizes = [max(1, int(round((n - 1) * f / tot))) for f in fracs]
    for m in sizes:
        parent.append(0)
        b = len(parent) - 1
        for _ in range(m - 1):
            if len(parent) < n:
                parent.append(b)
    while len(parent) < n:
        parent.append(0)
    return parent[:n]


def split_chains(n, k):
    parent = [-1]
    per = (n - 1) // k
    for _ in range(k):
        prev = 0
        for _ in range(per):
            if len(parent) < n:
                parent.append(prev)
                prev = len(parent) - 1
    while len(parent) < n:
        parent.append(0)
    return parent[:n]


def split_of(n, k, branch_builder):
    """root + k branches, each branch built by branch_builder(size) as a parent array (local root 0)."""
    parent = [-1]
    per = (n - 1) // k
    for _ in range(k):
        sub = branch_builder(per)
        base = len(parent)
        for v, p in enumerate(sub):
            if len(parent) >= n:
                break
            parent.append(0 if p < 0 else base + p)
    while len(parent) < n:
        parent.append(0)
    return parent[:n]


def hub_at_depth(n, d):
    parent = [-1]
    prev = 0
    for _ in range(d):
        parent.append(prev)
        prev = len(parent) - 1
    hub = prev
    while len(parent) < n:
        parent.append(hub)
    return parent


def spindle(n, pre, nleaf):
    """chain of `pre` from the root, a star of `nleaf` leaves at its end, then a chain to fill n."""
    parent = [-1]
    prev = 0
    for _ in range(pre):
        parent.append(prev)
        prev = len(parent) - 1
    mid = prev
    for _ in range(nleaf):
        if len(parent) < n:
            parent.append(mid)
    p2 = mid
    while len(parent) < n:
        parent.append(p2)
        p2 = len(parent) - 1
    return parent[:n]


def comb(n, teeth_len):
    parent = [-1]
    spine = [0]
    while len(parent) < n:
        parent.append(spine[-1])
        spine.append(len(parent) - 1)
        prev = spine[-1]
        for _ in range(teeth_len):
            if len(parent) >= n:
                break
            parent.append(prev)
            prev = len(parent) - 1
    return parent[:n]


def broom(n, exponent=0.5):
    m = min(math.ceil(n ** exponent), n - 1)
    parent = [-1]
    prev = 0
    for _ in range(m):
        parent.append(prev)
        prev = len(parent) - 1
    tip = prev
    while len(parent) < n:
        parent.append(tip)
    return parent


def double_broom(n, m):
    parent = [-1]
    prev = 0
    spine = [0]
    for _ in range(m):
        parent.append(prev)
        prev = len(parent) - 1
        spine.append(prev)
    i = 0
    while len(parent) < n:
        parent.append(spine[0] if i % 2 == 0 else spine[-1])
        i += 1
    return parent[:n]


def power_profile(n, m, p):
    sizes = [max(1, int(n * (1.0 - (j / m) ** p))) for j in range(m)]
    parent = [-1]
    spine = [0]
    prev = 0
    for _ in range(m - 1):
        parent.append(prev)
        prev = len(parent) - 1
        spine.append(prev)
    hang = [max(0, int(sizes[j] - sizes[j + 1] - 1)) for j in range(m - 1)] + [max(0, int(sizes[-1]) - 1)]
    for j, h in enumerate(hang):
        for _ in range(h):
            if len(parent) < n:
                parent.append(spine[j])
    while len(parent) < n:
        parent.append(spine[0])
    return parent[:n]


def profile_tree(n, sizes):
    """Pure spine with prescribed decreasing subtree sizes; the drops are hung as LEAVES.

    sizes: strictly decreasing ints in [1, n-1] (subtree sizes of the spine vertices below the root).
    Realizable iff consecutive drops are >= 1.  Leftover vertices are hung at the top spine vertex."""
    sizes = sorted({int(s) for s in sizes if 1 <= s <= n - 1}, reverse=True)
    parent = [-1]
    spine = []
    prev = 0
    for idx, target in enumerate(sizes):
        parent.append(prev)
        v = len(parent) - 1
        spine.append(v)
        prev = v
    # now every spine vertex v_idx must reach subtree size sizes[idx]
    # current subtree size of v_idx counting only the spine below it = len(sizes)-idx
    for idx in range(len(sizes)):
        have = len(sizes) - idx
        nxt_have = len(sizes) - idx - 1
        need = sizes[idx] - (sizes[idx + 1] if idx + 1 < len(sizes) else 0) - 1
        for _ in range(max(0, need)):
            if len(parent) < n:
                parent.append(spine[idx])
    while len(parent) < n:
        parent.append(spine[0] if spine else 0)
    return parent[:n]


def rrt(n, rng):
    return [-1] + [int(rng.integers(0, i)) for i in range(1, n)]


def ba_tree(n, rng):
    parent = [-1]
    targets = [0]
    for i in range(1, n):
        p = int(targets[int(rng.integers(0, len(targets)))])
        parent.append(p)
        targets.append(p)
        targets.append(i)
    return parent


def gw_tree(n_cap, mean, rng):
    parent = [-1]
    frontier = [0]
    while frontier and len(parent) < n_cap:
        v = frontier.pop(0)
        for _ in range(int(rng.poisson(mean))):
            if len(parent) >= n_cap:
                break
            parent.append(v)
            frontier.append(len(parent) - 1)
    return parent


def kesten_trunc(n, rng, mean=1.0):
    parent = [-1]
    spine = [0]
    while len(parent) < n:
        parent.append(spine[-1])
        spine.append(len(parent) - 1)
        attach = spine[-2]
        for _ in range(int(rng.poisson(mean))):
            if len(parent) >= n:
                break
            parent.append(attach)
            fr = [len(parent) - 1]
            while fr and len(parent) < n:
                x = fr.pop(0)
                for _ in range(int(rng.poisson(mean))):
                    if len(parent) >= n:
                        break
                    parent.append(x)
                    fr.append(len(parent) - 1)
    return parent[:n]


def uniform_shape(n, rng):
    return [-1] + [int(rng.integers(0, i)) for i in range(1, n)]
