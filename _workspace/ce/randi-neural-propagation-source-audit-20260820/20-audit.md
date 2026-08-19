# Stable pre-acquisition audit

Status: COMPLETE

Gate: PASS

Audited: 2026-08-20

## Stable snapshot

| File | SHA-256 |
|---|---|
| `00-contract.md` | `ec1fda064c2183bd9b1748605cce3749e585612bad051a5244fbf6aa404a8a50` |
| `10-sources.md` | `c9ae2305b499d83e1625d9bf38d613e06398130256cc0ce1aa6ebd81a4c85057` |
| `11-math.md` | `d72719713eb6d2103ef5fd6e54aa794e57642591c595c7da007f84e55196b0d2` |
| `12-routes.md` | `cf5581bd08ecc8c00afb92500c796ca195fd230477820c729bf46dbef6a9b704` |
| `artifacts/fetch_e2syt_public_manifest.py` | `7c63ae8b3554d651d65be547d84dcde4a86aa3a654ea1806dc1cb51110e68472` |
| `artifacts/e2syt-public-manifest.json` | `8ae094546532cc654dd6d49f3ffe5284f734598e7f6e05369a762579bba60e88` |

## Independent findings

The metadata receipt independently reproduces 223 assets, 113 subjects,
113 full assets, 110 segmentation assets and total size
`4,073,427,051,047` bytes.  Its normalized asset-list SHA-256 is
`ff206a13191908e92167817c60644265066ba578d8a58ea1ad1011466dcb47d5`.
The deterministic minimum `(bytes, path)` segmentation exemplar is:

- path: `sub-24/sub-24_ses-20211102-101248_desc-segmentation_ophys+ogen.nwb`
- asset UUID: `d076d282-162a-4946-a1c8-68e72b6cce54`
- expected bytes: `1,273,970`
- expected SHA-256: `40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e`

The acquisition script executed successfully and its Python source compiled.
The causal audit remains sound: pre/post baseline is not a no-light arm;
autoresponse inclusion is post-treatment selection; direct/monosynaptic
connectivity, endogenous `do(A)`, and `G -> tau` mediation remain blocked.

## Authorization

The implementation lane may:

1. download only the selected 1,273,970-byte segmentation exemplar;
2. verify its exact byte count and published SHA-256;
3. inspect only NWB/HDF5 schema and field presence.

It may not compute neural responses, pair effects, routing scores, Fisher
metrics, intervention effects, or select endpoints.  Any byte/hash mismatch or
reader ambiguity is an immediate STOP, not a reason to broaden acquisition.
