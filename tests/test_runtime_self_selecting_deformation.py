from reality_stone.clarus.runtime_self_selecting_deformation import SelfSelectingDeformationConfig, self_selecting_deformation

def test_m4r_reuses_one_candidate_bank_and_records_required_receipts() -> None:
    config = SelfSelectingDeformationConfig(dim=16, replay_epochs=1, rollout_horizon=2, seed=97401)
    result = self_selecting_deformation(97401, config)
    assert result["same_schedule_candidate_bank"]
    for task in ("loop8", "loop9"):
        row = result[task]
        assert len(row["candidate_bank"]) == 9
        assert row["snapshot_restore_parity"] and row["dense_sparse_parity"] and row["finite"]
        assert row["cutoff_audit"]["temporal_rows_after"] == 0
        assert row["cutoff_audit"]["hippocampal_rows_after"] == 0
        assert set(row["controls"]) == {"identity","no_selection","target_shuffled","trace_shuffled","sign_flipped"}
        epoch = row["epochs"][0]
        assert epoch["svd_rank_tol_1e-6"] >= 1
        assert len(epoch["candidates"]) == 9
        assert epoch["receipt"]["heldout_rows_used"] == 0
        assert row["factor_codebook_sha256"]
