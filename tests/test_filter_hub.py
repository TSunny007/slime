"""CPU unit tests for ``slime.rollout.filter_hub``.

Covers ``DynamicFilterOutput``, ``call_dynamic_filter``,
``MetricGatherer``, and the built-in ``check_reward_nonzero_std`` filter.
"""

from __future__ import annotations

import argparse

import pytest

from slime.rollout.filter_hub.base_types import DynamicFilterOutput, MetricGatherer, call_dynamic_filter
from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from slime.utils.types import Sample

NUM_GPUS = 0


# ---------------------------------------------------------------------------
# DynamicFilterOutput
# ---------------------------------------------------------------------------


class TestDynamicFilterOutput:
    def test_keep_defaults(self):
        out = DynamicFilterOutput(keep=True)
        assert out.keep is True
        assert out.reason is None

    def test_drop_with_reason(self):
        out = DynamicFilterOutput(keep=False, reason="low_quality")
        assert out.keep is False
        assert out.reason == "low_quality"


# ---------------------------------------------------------------------------
# call_dynamic_filter
# ---------------------------------------------------------------------------


class TestCallDynamicFilter:
    def test_none_fn_returns_keep(self):
        out = call_dynamic_filter(None)
        assert out.keep is True
        assert out.reason is None

    def test_fn_returning_output(self):
        def my_filter(*a, **kw):
            return DynamicFilterOutput(keep=False, reason="bad")

        out = call_dynamic_filter(my_filter)
        assert out.keep is False
        assert out.reason == "bad"

    def test_legacy_fn_returning_bool(self):
        def legacy_filter(*a, **kw):
            return True

        out = call_dynamic_filter(legacy_filter)
        assert out.keep is True


# ---------------------------------------------------------------------------
# MetricGatherer
# ---------------------------------------------------------------------------


class TestMetricGatherer:
    def test_empty_collect(self):
        mg = MetricGatherer()
        assert mg.collect() == {}

    def test_tracks_drop_reasons(self):
        mg = MetricGatherer()
        mg.on_dynamic_filter_drop("zero_std")
        mg.on_dynamic_filter_drop("zero_std")
        mg.on_dynamic_filter_drop("timeout")
        result = mg.collect()
        assert result["rollout/dynamic_filter/drop_zero_std"] == 2
        assert result["rollout/dynamic_filter/drop_timeout"] == 1

    def test_none_reason_ignored(self):
        mg = MetricGatherer()
        mg.on_dynamic_filter_drop(None)
        assert mg.collect() == {}

    def test_empty_string_reason_ignored(self):
        mg = MetricGatherer()
        mg.on_dynamic_filter_drop("")
        assert mg.collect() == {}


# ---------------------------------------------------------------------------
# check_reward_nonzero_std
# ---------------------------------------------------------------------------


class TestCheckRewardNonzeroStd:
    def _make_samples(self, rewards, reward_key=None):
        args = argparse.Namespace(reward_key=reward_key)
        samples = []
        for r in rewards:
            s = Sample()
            s.reward = r
            samples.append(s)
        return args, samples

    def test_diverse_rewards_kept(self):
        args, samples = self._make_samples([0.0, 0.5, 1.0])
        out = check_reward_nonzero_std(args, samples)
        assert bool(out.keep) is True
        assert out.reason is None

    def test_identical_rewards_dropped(self):
        args, samples = self._make_samples([0.5, 0.5, 0.5])
        out = check_reward_nonzero_std(args, samples)
        assert bool(out.keep) is False
        assert out.reason is not None
        assert "zero_std" in out.reason

    def test_near_zero_std_dropped(self):
        args, samples = self._make_samples([1.0, 1.0 + 1e-9, 1.0])
        out = check_reward_nonzero_std(args, samples)
        assert bool(out.keep) is False

    def test_with_reward_key(self):
        args = argparse.Namespace(reward_key="math")
        samples = [Sample(reward={"math": r}) for r in [0.0, 1.0, 0.5]]
        out = check_reward_nonzero_std(args, samples)
        assert bool(out.keep) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
