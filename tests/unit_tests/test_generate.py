import unittest
import pytest
import numpy as np
from open_clio.generate import (
    validate_hierarchy,
    generate_neighborhoods,
    get_contrastive_summaries,
)


class TestHierarchy:
    def test_invalid_hierarchy(self):
        with pytest.raises(ValueError):
            validate_hierarchy([], 10)
        with pytest.raises(ValueError):
            validate_hierarchy([1, 2, 3], 10)
        with pytest.raises(ValueError):
            validate_hierarchy([50], 10)
        with pytest.raises(ValueError):
            validate_hierarchy([101], 100)

    def test_valid_hierarchy(self):
        validate_hierarchy([3, 2, 1], 10)
        validate_hierarchy([12, 6], 100)
        validate_hierarchy([10], 10)  # boundary


class TestGetContrastiveSummaries:
    def test_selection(self):
        cluster_mask = np.array([True, True, False, False, False])
        embeddings = np.random.rand(5, 384)  # 5 examples, 384 dimensions
        summaries = [f"summary_{i}" for i in range(5)]

        result = get_contrastive_summaries(cluster_mask, embeddings, summaries)

        assert isinstance(result, list)
        assert len(result) <= 3  # 3 examples outside cluster
        for summary in result:
            assert summary in ["summary_2", "summary_3", "summary_4"]


class TestGenerateNeighborhoods:
    def test_generate_neighborhoods(self):
        cluster_embeddings = np.random.rand(10, 384)
        num_clusters = 10

        neighborhood_labels, k_neighborhoods = generate_neighborhoods(
            cluster_embeddings, num_clusters
        )

        assert len(neighborhood_labels) == 10
        assert k_neighborhoods >= 2
        assert k_neighborhoods <= 6
        assert len(np.unique(neighborhood_labels)) == k_neighborhoods


# propose clusters
# deduplicate clusters
# assign clusters
# rename clusters


# sth to do with level/csv creation
