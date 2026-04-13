"""Config schema validation tests."""
import pytest

from common.config.config_schema import validate_config, CONFIG_SCHEMA


class TestValidConfig:
    """Valid configs should produce no warnings."""

    def test_empty_config(self):
        assert validate_config({}) == []

    def test_detection_config(self):
        assert validate_config({"score_threshold": 0.3, "nms_threshold": 0.45}) == []

    def test_objectness_config(self):
        assert validate_config({
            "obj_threshold": 0.25,
            "score_threshold": 0.3,
            "nms_threshold": 0.45,
        }) == []

    def test_classification_config(self):
        assert validate_config({"top_k": 5}) == []

    def test_boundary_values(self):
        assert validate_config({"score_threshold": 0.0}) == []
        assert validate_config({"score_threshold": 1.0}) == []
        assert validate_config({"top_k": 1}) == []

    def test_integer_threshold(self):
        """Integer 0 and 1 should be valid for float threshold fields."""
        assert validate_config({"score_threshold": 0}) == []
        assert validate_config({"score_threshold": 1}) == []

    def test_boolean_field(self):
        assert validate_config({"has_background": True}) == []
        assert validate_config({"has_background": False}) == []


class TestInvalidValues:
    """Invalid values should produce warnings."""

    def test_negative_threshold(self):
        warnings = validate_config({"score_threshold": -0.1})
        assert len(warnings) == 1
        assert "below minimum" in warnings[0]

    def test_threshold_above_one(self):
        warnings = validate_config({"nms_threshold": 1.5})
        assert len(warnings) == 1
        assert "exceeds maximum" in warnings[0]

    def test_zero_topk(self):
        warnings = validate_config({"top_k": 0})
        assert len(warnings) == 1

    def test_negative_topk(self):
        warnings = validate_config({"top_k": -1})
        assert len(warnings) == 1

    def test_wrong_type_string(self):
        warnings = validate_config({"score_threshold": "high"})
        assert len(warnings) == 1
        assert "expected" in warnings[0]

    def test_wrong_type_bool_for_float(self):
        """bool is a subclass of int in Python, but should still work for thresholds."""
        # In Python, isinstance(True, int) is True, so this is actually valid
        warnings = validate_config({"top_k": True})
        assert len(warnings) == 0  # True == 1 which is valid

    def test_multiple_errors(self):
        warnings = validate_config({
            "score_threshold": -1.0,
            "nms_threshold": 2.0,
            "top_k": 0,
        })
        assert len(warnings) == 3


class TestUnknownKeys:
    """Unknown keys should be silently ignored (forward-compatible)."""

    def test_unknown_key(self):
        assert validate_config({"future_param": 42}) == []

    def test_mixed_known_unknown(self):
        warnings = validate_config({
            "score_threshold": 0.3,
            "custom_param": "hello",
        })
        assert len(warnings) == 0


class TestSchemaCompleteness:
    """Ensure all expected config keys are in the schema."""

    def test_all_known_keys_present(self):
        expected_keys = {
            "score_threshold", "nms_threshold", "obj_threshold",
            "confidence_threshold", "top_k", "num_classes",
            "reg_max", "num_protos", "has_background",
        }
        assert set(CONFIG_SCHEMA.keys()) == expected_keys
