# Source lane — C. elegans structural connectome

Status: COMPLETE

Limitation: source bytes are frozen locally, but redistribution permission is unresolved.

Recommended source: the OpenWorm project’s `CElegansNeuroML` repository, file [`herm_full_edgelist.csv`](https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/herm_full_edgelist.csv). This is a small machine-readable, project-authoritative materialization of the adult hermaphrodite wiring data and is preferable for an MVP replay to an interactive API or a spreadsheet.

| Evidence ID | Evidence | Source / exact object | Current status | Impact |
|---|---|---|---|---|
| S1 | Project repository and immutable commit identify the OpenWorm materialization. | [OpenWorm CElegansNeuroML](https://github.com/openworm/CElegansNeuroML/tree/b36380a36d2a6dda0f03c946c433524b25ea2268) | VERIFIED (accessed 2026-08-18) | Commit: `b36380a36d2a6dda0f03c946c433524b25ea2268`. |
| S2 | Frozen raw object is locally acquired: 252842 bytes, SHA-256 `0ab9baab5f404895b8dbeb8daa453c86e8f342961bc458cd19bf1b5f6a38d859`; header is `Source,Target,Weight,Type`. | [immutable raw CSV](https://raw.githubusercontent.com/openworm/CElegansNeuroML/b36380a36d2a6dda0f03c946c433524b25ea2268/herm_full_edgelist.csv) | VERIFIED | Local object: `artifacts/herm_full_edgelist.csv`. |
| S3 | OpenWorm’s dataset page says the Cook2019Herm dataset is extracted from SI5 adjacency matrices and lists 302 hermaphrodite neurons. | [Connectome Toolbox Cook2019Herm](https://openworm.org/ConnectomeToolbox/Loco3_Cook2019Herm_data/) | VERIFIED (accessed 2026-08-18) | Independent project documentation for population and extraction provenance. |
| S4 | Open Connectome Project documents 302 hermaphrodite neurons, chemical/gap-junction/NMJ classes, CSV/MAT/XLS access, and open CC licensing claim. | [OCP C. elegans page](https://openconnectomeproject.org/celegans) | VERIFIED as corroboration (accessed 2026-08-18) | Useful secondary project-authoritative cross-check; its API is not selected as the frozen byte object. |
| S5 | Published-scale context: OCP reports approximately 2,990 chemical synapses and 890 gap junctions for hermaphrodite; these are approximate and not acceptance values. | [OCP overview and facts](https://openconnectomeproject.org/celegans) | VERIFIED, informational only | Do not hard-code these rounded values; derive exact counts from frozen bytes. |
| S6 | Full tree at the frozen commit contains no LICENSE file. README public-domain language refers to VirtualWorm 3D/morphology context, not connectivity redistribution. | [repository tree](https://github.com/openworm/CElegansNeuroML/tree/b36380a36d2a6dda0f03c946c433524b25ea2268) | LIMITATION | Local research analysis only; redistribution permission is not established; raw CSV remains run-local. |
| S7 | Exact local metrics from the frozen bytes: 7379 rows; chemical 4681 / weight sum 27019; electrical 2698 / weight sum 12683; 448 endpoint IDs; 48 self-loops; 1359 normalized electrical pairs; 1339 reciprocal two-row pairs; 13 unequal reciprocal pairs; max 2 rows/electrical pair; 0 exact duplicate full rows. | Local acquisition artifact and hash manifest | VERIFIED | Exact metrics, not rounded published summaries. |

## Selection and semantics

The selected population is the adult hermaphrodite materialization represented by the file, not an assumed “302” acceptance constant. The CSV has four columns: `Source`, `Target`, `Weight`, and `Type`. `Source` is the presynaptic endpoint for chemical rows; `Target` is postsynaptic. `Weight` is the released connection multiplicity/weight and appears as an integer in the raw object. `Type` distinguishes at least `chemical` and `electrical`; electrical rows must be canonicalized as unordered pairs by the replay implementation and must not be doubled merely because a matrix/API exposes reciprocal entries. Rows can include muscles/other cells, so the exact included-cell population must be derived from the frozen file and declared by the implementation.

The frozen bytes are available locally and suitable for deterministic replay. This lane does not infer a redistribution license; the artifact is retained for local research analysis only.
