"""CPU unit tests for ``slime.utils.misc``.

Covers the pure-Python helpers that are testable without GPU, Ray, or
heavy external dependencies: ``SingletonMeta``, ``should_run_periodic_action``,
``Box``, ``group_by``, ``load_function``, and ``_chunk_by_size``.
"""

from __future__ import annotations

import pytest
import torch

from slime.utils.misc import (
    Box,
    SingletonMeta,
    _chunk_by_size,
    chunk_named_params_by_size,
    group_by,
    load_function,
    should_run_periodic_action,
)

NUM_GPUS = 0


# ---------------------------------------------------------------------------
# SingletonMeta
# ---------------------------------------------------------------------------


class TestSingletonMeta:
    def setup_method(self):
        SingletonMeta.clear_instances(SingletonMeta)

    def test_single_instance(self):
        class A(metaclass=SingletonMeta):
            pass

        a1 = A()
        a2 = A()
        assert a1 is a2

    def test_different_classes(self):
        class B(metaclass=SingletonMeta):
            pass

        class C(metaclass=SingletonMeta):
            pass

        assert B() is not C()

    def test_clear_instances(self):
        class D(metaclass=SingletonMeta):
            def __init__(self):
                self.val = id(self)

        d1 = D()
        SingletonMeta.clear_instances(D)
        d2 = D()
        assert d1 is not d2


# ---------------------------------------------------------------------------
# should_run_periodic_action
# ---------------------------------------------------------------------------


class TestShouldRunPeriodicAction:
    def test_none_interval_returns_false(self):
        assert should_run_periodic_action(0, interval=None) is False

    def test_on_interval(self):
        # step = rollout_id + 1 = 10, interval = 5 -> True
        assert should_run_periodic_action(9, interval=5) is True

    def test_off_interval(self):
        assert should_run_periodic_action(8, interval=5) is False

    def test_epoch_boundary(self):
        # step = 7, interval = 10, but num_rollout_per_epoch = 7 -> True
        assert should_run_periodic_action(6, interval=10, num_rollout_per_epoch=7) is True

    def test_last_rollout(self):
        assert should_run_periodic_action(4, interval=100, num_rollout=5) is True

    def test_not_last_rollout(self):
        assert should_run_periodic_action(3, interval=100, num_rollout=5) is False


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


class TestBox:
    def test_wrap_unwrap(self):
        b = Box(42)
        assert b.inner == 42

    def test_wrap_none(self):
        b = Box(None)
        assert b.inner is None


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------


class TestGroupBy:
    def test_basic_grouping(self):
        result = group_by([1, 2, 3, 4, 5], key=lambda x: x % 2)
        assert result == {1: [1, 3, 5], 0: [2, 4]}

    def test_no_key(self):
        result = group_by(["a", "b", "a", "c"])
        assert result == {"a": ["a", "a"], "b": ["b"], "c": ["c"]}

    def test_empty(self):
        assert group_by([], key=lambda x: x) == {}

    def test_unsorted_input(self):
        result = group_by([3, 1, 2, 1, 3], key=lambda x: x)
        assert result[1] == [1, 1]
        assert result[3] == [3, 3]


# ---------------------------------------------------------------------------
# load_function
# ---------------------------------------------------------------------------


class TestLoadFunction:
    def test_load_builtin(self):
        fn = load_function("os.path.join")
        import os.path

        assert fn is os.path.join

    def test_load_nonexistent_raises(self):
        with pytest.raises((ModuleNotFoundError, AttributeError)):
            load_function("nonexistent_module.func")


# ---------------------------------------------------------------------------
# _chunk_by_size / chunk_named_params_by_size
# ---------------------------------------------------------------------------


class TestChunkBySize:
    def test_single_chunk(self):
        items = [(1, 10), (2, 20), (3, 30)]
        chunks = list(_chunk_by_size(items, compute_size=lambda x: x[1], chunk_size=100))
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_multiple_chunks(self):
        items = [(1, 30), (2, 30), (3, 30), (4, 30)]
        # bucket grows: 30, 60, then 60+30=90 < 100 => 90, then 90+30=120 >= 100 => yield [1,2,3]
        chunks = list(_chunk_by_size(items, compute_size=lambda x: x[1], chunk_size=100))
        assert len(chunks) == 2
        assert chunks[0] == [(1, 30), (2, 30), (3, 30)]
        assert chunks[1] == [(4, 30)]

    def test_empty(self):
        chunks = list(_chunk_by_size([], compute_size=lambda x: x, chunk_size=10))
        assert chunks == []

    def test_item_exceeds_chunk_size(self):
        items = [(1, 200)]
        chunks = list(_chunk_by_size(items, compute_size=lambda x: x[1], chunk_size=100))
        assert len(chunks) == 1
        assert chunks[0] == [(1, 200)]

    def test_chunk_named_params(self):
        params = [
            ("w1", torch.zeros(10)),
            ("w2", torch.zeros(10)),
            ("w3", torch.zeros(10)),
        ]
        # Each tensor is 10 * 4 = 40 bytes (float32). chunk_size = 100 -> bucket
        # grows: 40, then 40+40=80 < 100, then 80+40=120 >= 100 -> yield [w1,w2], then [w3].
        chunks = list(chunk_named_params_by_size(params, chunk_size=100))
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
