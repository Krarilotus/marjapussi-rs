from ml.benchmark_four_model_runtime import summarize_timings


def test_summarize_timings_handles_p95_and_avgs():
    summary = summarize_timings(
        belief_ms=[1.0, 2.0, 3.0],
        decision_ms=[4.0, 5.0, 6.0],
        total_ms=[5.0, 7.0, 9.0],
    )
    assert summary.total_examples == 3
    assert summary.belief_ms_avg == 2.0
    assert summary.decision_ms_avg == 5.0
    assert summary.end_to_end_ms_avg == 7.0
    assert summary.end_to_end_ms_p95 == 9.0
