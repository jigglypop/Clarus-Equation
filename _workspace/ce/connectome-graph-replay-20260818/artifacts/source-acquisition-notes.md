# Acquisition notes

Status: COMPLETE (acquisition and hash verified; redistribution license unresolved).

Access date: 2026-08-18 (Asia/Seoul).

Frozen acquisition:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/openworm/CElegansNeuroML/b36380a36d2a6dda0f03c946c433524b25ea2268/herm_full_edgelist.csv" -OutFile artifacts/herm_full_edgelist.csv
Get-FileHash artifacts/herm_full_edgelist.csv -Algorithm SHA256
```

Frozen object: 252842 bytes; SHA-256 `0ab9baab5f404895b8dbeb8daa453c86e8f342961bc458cd19bf1b5f6a38d859`; commit `b36380a36d2a6dda0f03c946c433524b25ea2268`; header `Source,Target,Weight,Type`. Exact metrics: 7379 data rows; chemical 4681 rows / weight sum 27019; electrical 2698 / weight sum 12683; union endpoint IDs 448; self-loops 48; normalized electrical unordered pairs 1359; 1339 reciprocal two-row pairs; 13 reciprocal pairs with unequal weights; maximum two rows per electrical pair; zero exact duplicate full rows.

OpenWorm’s repository page exposes `herm_full_edgelist.csv` and identifies the project as an OpenWorm C. elegans model. The Connectome Toolbox independently identifies its Cook2019Herm materialization as extracted from Cook et al. 2019 SI5 adjacency matrices. The Open Connectome Project corroborates adult-hermaphrodite scope (302 neurons) and reports open CC terms, but its approximate summary counts are informational only.

The full tree has no LICENSE file. The README's public-domain statement concerns VirtualWorm 3D/morphology context and is not evidence that connectivity data may be redistributed. Therefore use is local research analysis only; redistribution is not established and the raw CSV remains run-local.
