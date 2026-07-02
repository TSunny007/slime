"""CPU unit tests for ``slime.utils.seqlen_balancing``.

Covers the partitioning algorithms (Karmarkar-Karp, greedy, bin-packing)
used for sequence-length-balanced data distribution across DP ranks and
microbatches.
"""

from __future__ import annotations

import pytest

from slime.utils.seqlen_balancing import (
    _split_bin_by_tokens,
    expand_bins_by_splitting,
    first_fit_pack,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    greedy_partition,
    karmarkar_karp,
)

NUM_GPUS = 0


def _partition_sums(seqlen_list, partitions):
    return [sum(seqlen_list[i] for i in p) for p in partitions]


def _all_indices_covered(partitions, n):
    flat = sorted(i for p in partitions for i in p)
    return flat == list(range(n))


# ---------------------------------------------------------------------------
# karmarkar_karp
# ---------------------------------------------------------------------------


class TestKarmarkarKarp:
    def test_uniform_lengths(self):
        seqlens = [100, 100, 100, 100]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=True)
        assert len(parts) == 2
        assert _all_indices_covered(parts, 4)
        sums = _partition_sums(seqlens, parts)
        assert sums[0] == sums[1] == 200

    def test_unequal_lengths_balance(self):
        seqlens = [10, 20, 30, 40, 50, 60]
        parts = karmarkar_karp(seqlens, k_partitions=3, equal_size=True)
        assert len(parts) == 3
        assert _all_indices_covered(parts, 6)
        sums = _partition_sums(seqlens, parts)
        spread = max(sums) - min(sums)
        assert spread <= 20

    def test_equal_size_constraint(self):
        seqlens = [1, 2, 3, 4, 5, 6]
        parts = karmarkar_karp(seqlens, k_partitions=3, equal_size=True)
        sizes = [len(p) for p in parts]
        assert all(s == 2 for s in sizes)

    def test_variable_size(self):
        seqlens = [1, 2, 3, 4, 5]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=False)
        assert len(parts) == 2
        assert _all_indices_covered(parts, 5)


# ---------------------------------------------------------------------------
# greedy_partition
# ---------------------------------------------------------------------------


class TestGreedyPartition:
    def test_basic_partition(self):
        seqlens = [10, 20, 30, 40]
        parts = greedy_partition(seqlens, k_partitions=2, equal_size=True)
        assert len(parts) == 2
        assert _all_indices_covered(parts, 4)

    def test_equal_size(self):
        seqlens = [5, 10, 15, 20, 25, 30]
        parts = greedy_partition(seqlens, k_partitions=3, equal_size=True)
        sizes = [len(p) for p in parts]
        assert all(s == 2 for s in sizes)

    def test_variable_size(self):
        seqlens = [1, 100, 2, 3]
        parts = greedy_partition(seqlens, k_partitions=2, equal_size=False)
        assert _all_indices_covered(parts, 4)


# ---------------------------------------------------------------------------
# get_seqlen_balanced_partitions (main entry point)
# ---------------------------------------------------------------------------


class TestGetSeqlenBalancedPartitions:
    def test_basic(self):
        seqlens = [10, 20, 30, 40]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)
        assert len(parts) == 2
        assert _all_indices_covered(parts, 4)
        # each partition is internally sorted
        for p in parts:
            assert p == sorted(p)

    def test_single_partition(self):
        seqlens = [5, 10, 15]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=1, equal_size=False)
        assert len(parts) == 1
        assert sorted(parts[0]) == [0, 1, 2]

    def test_too_few_items_raises(self):
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions([1], k_partitions=2, equal_size=False)


# ---------------------------------------------------------------------------
# first_fit_pack
# ---------------------------------------------------------------------------


class TestFirstFitPack:
    def test_all_fit_in_one_bin(self):
        bins = first_fit_pack([10, 20, 30], max_tokens_per_bin=100)
        assert len(bins) == 1
        assert sorted(bins[0]) == [0, 1, 2]

    def test_each_item_separate(self):
        bins = first_fit_pack([50, 50, 50], max_tokens_per_bin=50)
        assert len(bins) == 3

    def test_mixed_packing(self):
        bins = first_fit_pack([30, 20, 50, 10], max_tokens_per_bin=60)
        flat = sorted(i for b in bins for i in b)
        assert flat == [0, 1, 2, 3]
        for b in bins:
            assert sum([30, 20, 50, 10][i] for i in b) <= 60

    def test_oversized_item_gets_own_bin(self):
        bins = first_fit_pack([100, 10, 10], max_tokens_per_bin=50)
        # the oversized item (100) goes alone
        oversized = [b for b in bins if 0 in b]
        assert len(oversized) == 1
        assert oversized[0] == [0]


# ---------------------------------------------------------------------------
# _split_bin_by_tokens
# ---------------------------------------------------------------------------


class TestSplitBinByTokens:
    def test_basic_split(self):
        lengths = [10, 20, 30, 40]
        halves = _split_bin_by_tokens([0, 1, 2, 3], lengths)
        assert len(halves) == 2
        flat = sorted(halves[0] + halves[1])
        assert flat == [0, 1, 2, 3]

    def test_split_preserves_all_indices(self):
        lengths = [5, 5, 5, 5, 5, 5]
        halves = _split_bin_by_tokens([0, 1, 2, 3, 4, 5], lengths)
        flat = sorted(halves[0] + halves[1])
        assert flat == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# expand_bins_by_splitting
# ---------------------------------------------------------------------------


class TestExpandBinsBySplitting:
    def test_expand_to_target(self):
        bins = [[0, 1, 2, 3]]
        lengths = [10, 20, 30, 40]
        expand_bins_by_splitting(bins, target_count=3, lengths=lengths)
        flat = sorted(i for b in bins for i in b)
        assert flat == [0, 1, 2, 3]
        assert len(bins) >= 3

    def test_cannot_exceed_singletons(self):
        bins = [[0], [1], [2]]
        lengths = [10, 20, 30]
        expand_bins_by_splitting(bins, target_count=5, lengths=lengths)
        assert len(bins) == 3  # can't split singletons

    def test_already_at_target(self):
        bins = [[0], [1]]
        lengths = [10, 20]
        expand_bins_by_splitting(bins, target_count=2, lengths=lengths)
        assert len(bins) == 2


# ---------------------------------------------------------------------------
# get_reverse_idx
# ---------------------------------------------------------------------------


class TestGetReverseIdx:
    def test_identity(self):
        assert get_reverse_idx([0, 1, 2]) == [0, 1, 2]

    def test_permutation(self):
        idx_map = [2, 0, 1]
        rev = get_reverse_idx(idx_map)
        assert rev[idx_map[0]] == 0
        assert rev[idx_map[1]] == 1
        assert rev[idx_map[2]] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
