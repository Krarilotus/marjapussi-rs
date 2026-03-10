from ml.behavior_score import BehaviorEvalSummary, compute_behavior_score


def test_behavior_score_penalizes_pass_collapse():
    healthy = compute_behavior_score(
        BehaviorEvalSummary(
            point_diff=5.0,
            pass_game_rate=0.02,
            avg_bid=145.0,
            contract_made_rate=0.55,
            trump_call_rate=0.75,
        )
    )
    collapsed = compute_behavior_score(
        BehaviorEvalSummary(
            point_diff=10.0,
            pass_game_rate=0.95,
            avg_bid=121.0,
            contract_made_rate=0.05,
            trump_call_rate=0.05,
        )
    )
    assert healthy.total > collapsed.total


def test_behavior_score_penalizes_minimal_bid_behavior():
    low_bid = compute_behavior_score(
        BehaviorEvalSummary(
            point_diff=0.0,
            pass_game_rate=0.0,
            avg_bid=121.0,
            contract_made_rate=0.40,
            trump_call_rate=0.60,
        )
    )
    normal_bid = compute_behavior_score(
        BehaviorEvalSummary(
            point_diff=0.0,
            pass_game_rate=0.0,
            avg_bid=145.0,
            contract_made_rate=0.40,
            trump_call_rate=0.60,
        )
    )
    assert normal_bid.total > low_bid.total
