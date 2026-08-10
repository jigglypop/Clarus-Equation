from __future__ import annotations

from reality_stone.clarus.fusion_sciencedb_payload_loop import (
    current_sciencedb_v1_payload_audit,
)


def main() -> None:
    audit = current_sciencedb_v1_payload_audit()

    print("FUSION SCIENCEDB V1 LOCAL PAYLOAD GATE")
    print(f" source dataset                  {audit.source_dataset_doi}")
    print(f" dataset id                      {audit.source_dataset_id}")
    print(f" source version                  {audit.source_dataset_version}")
    print(f" source license                  {audit.source_dataset_license}")
    print(f" exact files                     {audit.runtime_entry_count}/6")
    print(f" exact bytes                     {audit.runtime_total_bytes}/8602")
    print(f" raw MD5/SHA-256 match           {audit.all_file_hashes_match}")
    print(f" parsed table structures         {audit.all_table_structures_match}")
    print(f" scalar CS+ERR tables            {len(audit.scalar_err_table_names)}")
    print(f" Legendre A1..A12 tables         {len(audit.legendre_a1_a12_table_names)}")
    print(
        " numeric covariance matrix       "
        f"{audit.numeric_covariance_matrix_or_correlation_payload_available}"
    )
    print(
        " initial-state spin operator      "
        f"{audit.initial_state_spin_columns_or_operator_available}"
    )
    print(f" payload integrity               {audit.payload_integrity_gate_pass}")
    print(
        " physical polarized reaction     "
        f"{audit.physical_polarized_reaction_evidence_gate_pass}"
    )
    print(f" maximum supported stage         {audit.maximum_supported_stage}")

    if not audit.payload_integrity_gate_pass:
        raise RuntimeError("ScienceDB V1 local payload identity or table structure regressed")
    if audit.numeric_covariance_matrix_or_correlation_payload_available:
        raise RuntimeError("unexpected covariance promotion requires independent review")
    if audit.initial_state_spin_columns_or_operator_available:
        raise RuntimeError("unexpected initial-state spin promotion requires independent review")
    if audit.physical_polarized_reaction_evidence_gate_pass:
        raise RuntimeError("unexpected polarized-reaction promotion requires independent review")


if __name__ == "__main__":
    main()
