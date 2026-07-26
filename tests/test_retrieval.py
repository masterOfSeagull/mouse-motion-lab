from __future__ import annotations

import numpy as np

from mouselearn.models import RetrievalConfig, RetrievalGenerator


def test_retrieval_is_seeded_and_flags_out_of_distribution_conditions() -> None:
    model = RetrievalGenerator(RetrievalConfig(neighbor_count=2, temperature=0.1, maximum_neighbor_distance=1.0)).fit(
        np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 9.0]]), np.array([[1.0], [2.0], [3.0]]),
    )
    first = model.generate(np.array([0.05, 0.0]), seed=7)
    second = model.generate(np.array([0.05, 0.0]), seed=7)
    assert first.source_index == second.source_index
    assert np.array_equal(first.output, second.output)
    assert not first.out_of_distribution
    assert model.generate(np.array([100.0, 100.0]), seed=7).out_of_distribution
