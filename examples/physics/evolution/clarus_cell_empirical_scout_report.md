# Clarus cell empirical scout

- passed: `True`
- local empirical data ready: `True`
- first gate report exists: `True`
- first empirical gate: `clarus_cell_crisprbrain_neuron_maintenance_gate.py`
- parallel proliferative gate: `clarus_cell_depmap_operator_dependency_gate.py`
- architecture gate: `clarus_cell_jump_morphology_operator_gate.py`

## local empirical files

- `crisprbrain_hit_class`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\41593_2021_862_MOESM4_hit_class.csv`
- `psapko_rnaseq`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\GSE152988_WT_vs_PSAPKO.csv.gz`
- `depmap_operator_subset`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\depmap_24q4_clarus_operator_dependency_subset.csv`
- `microglia_support_tables`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\41593_2022_1131_microglia_supp_tables.xlsx`
- `astrocyte_screen_table`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\41593_2022_1180_astrocyte_screen_table2.xlsx`
- `astrocyte_cropseq_table`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\41593_2022_1180_astrocyte_cropseq_table6.xlsx`
- `hpa_subcellular_location`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\hpa_subcellular_location.tsv.zip`
- `jump_profile_index`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_profile_index_v0.11.0.json`
- `jump_crispr_profiles_pca_corrected`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_profiles_pca_corrected.parquet`
- `jump_crispr_profiles_interpretable`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_profiles_interpretable.parquet`
- `jump_operator_morphology_subset`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_clarus_operator_morphology_subset.csv`
- `jump_mito_direct_feature_subset`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_mito_direct_features.parquet`
- `jump_mito_direct_gene_summary`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_crispr_mito_direct_gene_summary.csv`
- `jump_compound_mito_direct_feature_subset`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_compound_mito_direct_features.parquet`
- `jump_compound_mito_positive_control_summary`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\jump_compound_mito_positive_control_summary.csv`
- `replogle_k562_essential_normalized_bulk`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\replogle_k562_essential_normalized_bulk_01.h5ad`
- `replogle_rpe1_normalized_bulk`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\replogle_rpe1_normalized_bulk_01.h5ad`
- `replogle_perturbseq_clarus_state_summary`: `True` at `C:\Users\dongh\OneDrive\Desktop\Clarus-Equation\data\evolution\clarus_cell\replogle_perturbseq_clarus_state_summary.csv`

## ranked sources

| rank | source | form | operators | perturb | conditioned | priority | next gate |
|---:|---|---|---|---|---|---:|---|
| 1 | [BioGRID ORCS CRISPR screen index](https://orcs.thebiogrid.org/) | many organisms, cell lines, and phenotypes | `B,U,E,A,I,D,Q,R` | `True` | `True` | 1.209 | `clarus_cell_orcs_screen_triage_gate.py` |
| 2 | [CRISPRbrain human iPSC-derived neuron screens](https://www.nature.com/articles/s41593-021-00862-0) | human postmitotic iPSC-derived neurons | `E,A,D,Q,R,S` | `True` | `True` | 1.107 | `clarus_cell_crisprbrain_neuron_maintenance_gate.py` |
| 3 | [Genome-scale Perturb-seq in K562 and RPE1](https://pubmed.ncbi.nlm.nih.gov/35688146/) | human proliferative K562 and RPE1 cells | `E,A,I,D,Q,R` | `True` | `True` | 0.987 | `clarus_cell_perturbseq_state_reconstruction_gate.py` |
| 4 | [DepMap Project Achilles CRISPR gene effect](https://depmap.org/portal/achilles/) | human proliferative/cancer cell lines | `E,A,I,U,D,Q,R` | `True` | `False` | 0.978 | `clarus_cell_depmap_operator_dependency_gate.py` |
| 5 | [JUMP Cell Painting genetic and chemical perturbation profiles](https://github.com/jump-cellpainting/datasets) | human U2OS high-content imaging | `B,U,E,D,Q,R` | `True` | `False` | 0.867 | `clarus_cell_jump_morphology_operator_gate.py` |
| 6 | [Model protocell membrane growth and division experiments](https://pubs.acs.org/doi/10.1021/ja900919c) | fatty-acid model protocells | `B,U,I,R` | `True` | `True` | 0.824 | `clarus_cell_protocell_boundary_recurrence_gate.py` |
| 7 | [Nonenzymatic RNA synthesis inside fatty-acid vesicles](https://www.science.org/doi/10.1126/science.1241888) | model RNA protocells | `B,U,I,D` | `True` | `True` | 0.824 | `clarus_cell_protocell_template_copying_gate.py` |
| 8 | [CRISPRbrain human microglia and astrocyte screens](https://www.nature.com/articles/s41593-022-01131-4) | human iPSC-derived glia | `D,Q,S,R` | `True` | `True` | 0.764 | `clarus_cell_glia_support_operator_gate.py` |
| 9 | [Human Protein Atlas subcellular and brain single-cell resources](https://www.proteinatlas.org/humanproteome/subcellular) | human cells, tissues, and brain cell types | `B,U,A,I,Q,S` | `False` | `False` | 0.667 | `clarus_cell_hpa_operator_gene_set_gate.py` |
| 10 | [OpenCell endogenous tagging map](https://pubmed.ncbi.nlm.nih.gov/35271311/) | human HEK293T-derived live-cell protein localization | `B,U,A,I,Q` | `False` | `False` | 0.556 | `clarus_cell_opencell_operator_blueprint_gate.py` |

## operator-to-data reading

- `B` boundary: strongest empirical route is JUMP morphology plus protocell vesicle assays.
- `U` regulated ports/traffic: strongest route is JUMP morphology plus OpenCell/HPA localization.
- `E/A` energy and metabolism: strongest route is DepMap, CRISPRbrain neuron survival, and OXPHOS screens via ORCS.
- `I` identity template: strongest route is Perturb-seq state reconstruction and DepMap essentiality.
- `D/Q` damage and repair: strongest route is CRISPRbrain oxidative-stress/lysosome/autophagy screens.
- `S` support context: strongest route is glia CRISPRbrain plus HPA/brain cell-type atlases.
- `R` recurrence: proliferative branch uses DepMap fitness; postmitotic branch uses neuron survival/maintenance.

## recommended next gates

1. `clarus_cell_crisprbrain_neuron_maintenance_gate.py`: test postmitotic neural maintenance.
2. `clarus_cell_depmap_operator_dependency_gate.py`: test proliferative recurrence dependencies.
3. `clarus_cell_jump_morphology_operator_gate.py`: test morphology/operator separation.
4. `clarus_cell_protocell_boundary_recurrence_gate.py`: test primitive boundary/template recurrence.

## claim boundary

This scout only identifies empirical routes.  Promotion decisions belong to the operator-level gate reports.
