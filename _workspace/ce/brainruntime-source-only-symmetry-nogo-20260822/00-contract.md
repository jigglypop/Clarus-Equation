# BA-TR7: source-only uniform-substrate symmetry no-go

Status: COMPLETE

Mode: light

PREDECESSOR: `_workspace/ce/brainruntime-broad-edge-selector-20260822`

## Question

If the externally supplied $H(\text{payload})$ pulse is removed from BA-TR6, can a source-only experience identify four payload-preserving edges on the same uniform 32-edge substrate?

## Frozen probe

Use the BA-TR6 source snapshot with $W_{hs}=1$ for every $h\in H$ and $s\in S_0\cup S_1$. Reset all continuous and discrete runtime state. Pulse one source payload only at tick zero, never pulse $H$ or $Y$, and run through tick $L+1=3$ for delay $L=2$.

The old $0..L$ window is an explicit adverse receipt: it must contain no hidden arrival and zero candidate eligibility. The extended $0..L+1$ window must record the first real arrival.

Because the four hidden coordinates have identical activation dynamics and receive the same weight from every source coordinate,

$$
h_1(t)=h_2(t)=h_3(t)=h_4(t)
$$

at first arrival. For a source coordinate $s_k$, the four local values $E_{h_i s_k}$ are therefore equal. Averaging the four payload experiences for one source block produces sixteen tied positive candidate edges. The required four-edge selector has

$$
s_{(4)}-s_{(5)}=0
$$

and must abstain before any decoder or endpoint is opened.

## Development decision

Run seeds `97901..97916`; keep `99901..99932` sealed. The registered result is `SOURCE_ONLY_SYMMETRY_NO_GO` only if every seed shows zero hidden activity through tick $L$, nonzero and row-equal hidden activation at $L+1$, sixteen tied positive candidates per factor cue, zero top-four boundary gap, threshold-profile permutation invariance at first arrival, zero target/H/Y/decoder/endpoint reads, and `endpoint_opened=false`.

Any lexicographic/stable-sort selection, threshold retuning, added noise, changed horizon, nonuniform weight, or hidden pulse is a new mechanism and cannot rescue this test.

## Claim ceiling

This no-go establishes only that the current uniform source-only synthetic apparatus lacks information to identify a payload-preserving hidden coordinate. It does not prove that biological networks cannot self-organize. The next candidate must state an explicit local symmetry breaker, such as seeded heterogeneity plus competition/homeostasis, and test it as a new mechanism.
