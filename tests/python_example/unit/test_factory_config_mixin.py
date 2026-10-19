"""_FactoryConfigMixin.load_config must tolerate the factory's own config dict.

Concrete factories store their constructor argument by reference
(``self.config = config or {}``), so the ordinary shape
``f = SomeFactory(cfg); f.load_config(cfg)`` aliases ``self.config`` and the argument.
Inserting the score_threshold -> conf_threshold alias while iterating that same dict
raised ``RuntimeError: dictionary changed size during iteration``.

The trigger is conditional: it needs ``score_threshold`` present AND ``conf_threshold``
absent, so a config without an aliased key would pass even against the broken loop.
"""
from __future__ import annotations

from common.base.i_factory import _FactoryConfigMixin


class _Factory(_FactoryConfigMixin):
    """Minimal stand-in that stores config by reference, as every real factory does."""

    def __init__(self, config=None):
        self.config = config or {}


def test_load_config_accepts_the_factory_s_own_config_dict():
    config = {"score_threshold": 0.7}
    factory = _Factory(config)
    assert factory.config is config, "precondition: stored by reference"

    factory.load_config(config)

    assert factory.config["score_threshold"] == 0.7
    assert factory.config["conf_threshold"] == 0.7


def test_load_config_aliases_when_given_a_separate_dict():
    factory = _Factory({"nms_threshold": 0.45})
    factory.load_config({"score_threshold": 0.3})
    assert factory.config["conf_threshold"] == 0.3
    assert factory.config["nms_threshold"] == 0.45


def test_an_explicit_alias_value_is_not_overwritten():
    """`alias not in config` means a caller-supplied conf_threshold wins over the alias."""
    factory = _Factory()
    factory.load_config({"score_threshold": 0.7, "conf_threshold": 0.5})
    assert factory.config["conf_threshold"] == 0.5
