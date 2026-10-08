import torch

from paretojev.data.synthetic import SyntheticConfig, SyntheticSlateDataset, collate_slates
from paretojev.models.behavior import SASRecMMoEConfig, SASRecMMoERanker
from paretojev.models.pareto_reranker import ParetoJevConfig, ParetoJevReranker


def test_pareto_reranker_shapes() -> None:
    dataset = SyntheticSlateDataset(
        SyntheticConfig(n_slates=2, slate_size=5, n_objectives=3, feature_dim=7)
    )
    batch = collate_slates([dataset[0], dataset[1]])
    model = ParetoJevReranker(
        ParetoJevConfig(
            candidate_feature_dim=7,
            n_objectives=3,
            hidden_dim=24,
            n_heads=4,
            n_layers=1,
            ff_dim=48,
        )
    )
    output = model(batch)
    assert output["score_mean"].shape == (2, 5)
    assert output["objective_logits"].shape == (2, 5, 3)
    assert output["fusion_gate"].shape == (2, 5, 3)


def test_sasrec_mmoe_shapes() -> None:
    config = SASRecMMoEConfig(
        num_items=100,
        n_objectives=4,
        max_history_length=10,
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        n_experts=3,
        expert_dim=12,
    )
    model = SASRecMMoERanker(config)
    history = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    candidates = torch.tensor([[8, 9, 10], [11, 12, 13]])
    output = model.probabilities(history, candidates)
    assert output.shape == (2, 3, 4)
    assert torch.all((output >= 0) & (output <= 1))
