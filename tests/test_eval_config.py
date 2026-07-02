"""CPU unit tests for ``slime.utils.eval_config``.

Covers the helper functions and ``EvalDatasetConfig`` dataclass that
resolve per-dataset evaluation overrides.  These are pure-Python and
exercise the priority chain: dataset-level > defaults > args fallback.
"""

from __future__ import annotations

import argparse

import pytest

from slime.utils.eval_config import (
    EvalDatasetConfig,
    _apply_dataset_field_overrides,
    _first_not_missing,
    _pick_from_mapping,
    build_eval_dataset_configs,
    ensure_dataset_list,
    pick_from_args,
)

NUM_GPUS = 0

_MISSING = object()


# ---------------------------------------------------------------------------
# _first_not_missing
# ---------------------------------------------------------------------------


def _sentinel():
    """Return the module-private sentinel so tests can construct "missing" values."""
    from slime.utils.eval_config import _MISSING as M

    return M


class TestFirstNotMissing:
    def test_all_missing(self):
        M = _sentinel()
        assert _first_not_missing(M, M, M) is M

    def test_first_wins(self):
        M = _sentinel()
        assert _first_not_missing(42, M) == 42

    def test_second_wins(self):
        M = _sentinel()
        assert _first_not_missing(M, "hello") == "hello"

    def test_none_is_not_missing(self):
        M = _sentinel()
        assert _first_not_missing(None, M) is None


# ---------------------------------------------------------------------------
# _pick_from_mapping
# ---------------------------------------------------------------------------


class TestPickFromMapping:
    def test_key_present(self):
        assert _pick_from_mapping({"a": 1, "b": 2}, ("b",)) == 2

    def test_first_matching_key(self):
        assert _pick_from_mapping({"x": 10, "y": 20}, ("x", "y")) == 10

    def test_no_match_returns_missing(self):
        M = _sentinel()
        result = _pick_from_mapping({"a": 1}, ("z",))
        assert result is M

    def test_none_keys_returns_missing(self):
        M = _sentinel()
        result = _pick_from_mapping({"a": 1}, None)
        assert result is M


# ---------------------------------------------------------------------------
# pick_from_args
# ---------------------------------------------------------------------------


class TestPickFromArgs:
    def test_first_attr_found(self):
        args = argparse.Namespace(eval_temperature=0.7, rollout_temperature=1.0)
        assert pick_from_args(args, ("eval_temperature", "rollout_temperature")) == 0.7

    def test_fallback_to_second(self):
        args = argparse.Namespace(eval_temperature=None, rollout_temperature=0.9)
        assert pick_from_args(args, ("eval_temperature", "rollout_temperature")) == 0.9

    def test_missing_attrs_returns_none(self):
        args = argparse.Namespace()
        assert pick_from_args(args, ("no_such_attr",)) is None

    def test_all_none_returns_none(self):
        args = argparse.Namespace(a=None, b=None)
        assert pick_from_args(args, ("a", "b")) is None


# ---------------------------------------------------------------------------
# EvalDatasetConfig
# ---------------------------------------------------------------------------


class TestEvalDatasetConfig:
    def test_basic_construction(self):
        cfg = EvalDatasetConfig(name="math", path="/data/math.jsonl")
        assert cfg.name == "math"
        assert cfg.path == "/data/math.jsonl"
        assert cfg.metadata_overrides == {}

    def test_metadata_overrides_none_becomes_empty_dict(self):
        cfg = EvalDatasetConfig(name="x", path="p", metadata_overrides=None)
        assert cfg.metadata_overrides == {}

    def test_metadata_overrides_bad_type_raises(self):
        with pytest.raises(TypeError, match="metadata_overrides must be a mapping"):
            EvalDatasetConfig(name="x", path="p", metadata_overrides=[1, 2])

    def test_min_eval_samples_zero_raises(self):
        with pytest.raises(ValueError, match="min_eval_samples must be positive"):
            EvalDatasetConfig(name="x", path="p", min_eval_samples=0)

    def test_min_eval_samples_negative_raises(self):
        with pytest.raises(ValueError, match="min_eval_samples must be positive"):
            EvalDatasetConfig(name="x", path="p", min_eval_samples=-1)

    def test_cache_key_tuple(self):
        cfg = EvalDatasetConfig(name="a", path="b", input_key="q", label_key="l")
        assert cfg.cache_key == ("a", "b", "q", "l", None, None)

    def test_inject_metadata_adds_rm_type(self):
        cfg = EvalDatasetConfig(name="a", path="b", rm_type="math")
        result = cfg.inject_metadata({})
        assert result == {"rm_type": "math"}

    def test_inject_metadata_with_overrides(self):
        cfg = EvalDatasetConfig(name="a", path="b", metadata_overrides={"k": "v"})
        result = cfg.inject_metadata({"existing": 1})
        assert result == {"existing": 1, "k": "v"}

    def test_inject_metadata_non_dict_input(self):
        cfg = EvalDatasetConfig(name="a", path="b", rm_type="f1")
        result = cfg.inject_metadata("not-a-dict")
        assert result == {"rm_type": "f1"}


# ---------------------------------------------------------------------------
# ensure_dataset_list
# ---------------------------------------------------------------------------


class TestEnsureDatasetList:
    def test_none_returns_empty(self):
        assert ensure_dataset_list(None) == []

    def test_dict_input_inserts_name(self):
        result = ensure_dataset_list({"math": {"path": "/p"}, "code": {"path": "/c"}})
        assert len(result) == 2
        assert result[0]["name"] == "math"
        assert result[1]["name"] == "code"

    def test_dict_preserves_existing_name(self):
        result = ensure_dataset_list({"math": {"name": "custom_math", "path": "/p"}})
        assert result[0]["name"] == "custom_math"

    def test_list_input(self):
        result = ensure_dataset_list([{"name": "a", "path": "/a"}, {"name": "b", "path": "/b"}])
        assert len(result) == 2

    def test_list_missing_name_raises(self):
        with pytest.raises(ValueError, match="must include a `name` field"):
            ensure_dataset_list([{"path": "/a"}])

    def test_tuple_input(self):
        result = ensure_dataset_list(({"name": "a", "path": "/a"},))
        assert len(result) == 1

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="must be either a list or a mapping"):
            ensure_dataset_list(42)


# ---------------------------------------------------------------------------
# _apply_dataset_field_overrides
# ---------------------------------------------------------------------------


class TestApplyDatasetFieldOverrides:
    def _make_args(self, **kwargs):
        return argparse.Namespace(**kwargs)

    def test_dataset_level_wins(self):
        args = self._make_args(eval_temperature=0.5, rollout_temperature=1.0)
        cfg = {"temperature": 0.3}
        defaults = {"temperature": 0.7}
        spec = {
            "temperature": {
                "dataset_keys": ("temperature",),
                "default_keys": ("temperature",),
                "arg_attrs": ("eval_temperature", "rollout_temperature"),
            }
        }
        _apply_dataset_field_overrides(args, cfg, defaults, spec)
        assert cfg["temperature"] == 0.3

    def test_default_wins_when_dataset_missing(self):
        args = self._make_args(eval_temperature=0.5)
        cfg = {}
        defaults = {"temperature": 0.7}
        spec = {
            "temperature": {
                "dataset_keys": ("temperature",),
                "default_keys": ("temperature",),
                "arg_attrs": ("eval_temperature",),
            }
        }
        _apply_dataset_field_overrides(args, cfg, defaults, spec)
        assert cfg["temperature"] == 0.7

    def test_args_fallback(self):
        args = self._make_args(eval_temperature=0.5)
        cfg = {}
        defaults = {}
        spec = {
            "temperature": {
                "dataset_keys": ("temperature",),
                "default_keys": ("temperature",),
                "arg_attrs": ("eval_temperature",),
            }
        }
        _apply_dataset_field_overrides(args, cfg, defaults, spec)
        assert cfg["temperature"] == 0.5


# ---------------------------------------------------------------------------
# build_eval_dataset_configs
# ---------------------------------------------------------------------------


class TestBuildEvalDatasetConfigs:
    def test_basic_build(self):
        args = argparse.Namespace(
            n_samples_per_eval_prompt=None,
            n_samples_per_prompt=None,
            eval_temperature=None,
            rollout_temperature=None,
            eval_top_p=None,
            rollout_top_p=None,
            eval_top_k=None,
            rollout_top_k=None,
            eval_max_response_len=None,
            rollout_max_response_len=None,
            eval_input_key=None,
            input_key=None,
            eval_label_key=None,
            label_key=None,
            eval_tool_key=None,
            tool_key=None,
            metadata_key=None,
            multimodal_keys=None,
            apply_chat_template=None,
            apply_chat_template_kwargs=None,
            eval_custom_rm_path=None,
            custom_rm_path=None,
        )
        raw = [{"name": "gsm8k", "path": "/data/gsm8k.jsonl"}]
        configs = build_eval_dataset_configs(args, raw, defaults={})
        assert len(configs) == 1
        assert configs[0].name == "gsm8k"

    def test_defaults_applied(self):
        args = argparse.Namespace(
            n_samples_per_eval_prompt=None,
            n_samples_per_prompt=None,
            eval_temperature=None,
            rollout_temperature=None,
            eval_top_p=None,
            rollout_top_p=None,
            eval_top_k=None,
            rollout_top_k=None,
            eval_max_response_len=None,
            rollout_max_response_len=None,
            eval_input_key=None,
            input_key=None,
            eval_label_key=None,
            label_key=None,
            eval_tool_key=None,
            tool_key=None,
            metadata_key=None,
            multimodal_keys=None,
            apply_chat_template=None,
            apply_chat_template_kwargs=None,
            eval_custom_rm_path=None,
            custom_rm_path=None,
        )
        raw = [{"name": "test", "path": "/data/test.jsonl"}]
        defaults = {"temperature": 0.3}
        configs = build_eval_dataset_configs(args, raw, defaults)
        assert configs[0].temperature == 0.3

    def test_empty_raw_config(self):
        args = argparse.Namespace()
        configs = build_eval_dataset_configs(args, [], defaults={})
        assert configs == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
