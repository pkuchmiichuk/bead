"""Tests for :mod:`bead.config.protocol`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from bead.config.protocol import (
    AnchorSpec,
    DriftConfig,
    FamilySpec,
    ProtocolConfig,
    TemplateVariantSpec,
)
from bead.config.serialization import to_yaml
from bead.protocol import (
    AnnotationProtocol,
    ContextualTemplateRealization,
    LMRealization,
    SemanticAnchor,
    StructuralDriftValidator,
    TemplateRealization,
    register_context_predicate,
)
from bead.protocol.context import ProtocolContext


def _is_verb(ctx: ProtocolContext) -> bool:
    return ctx.target_upos == "VERB"


register_context_predicate("test_is_verb", _is_verb)


class TestAnchorSpec:
    """Tests for :class:`AnchorSpec`."""

    def test_build_minimal(self) -> None:
        spec = AnchorSpec(
            name="completion",
            target_property="telicity",
            canonical_prompt="Does [[situation]] end?",
            options=("no", "yes"),
            is_ordered=False,
            required_span_labels=frozenset({"situation"}),
        )
        anchor = spec.build()
        assert isinstance(anchor, SemanticAnchor)
        assert anchor.name == "completion"
        assert anchor.response_space.options == ("no", "yes")
        assert anchor.response_space.semantic_poles is None

    def test_build_with_poles(self) -> None:
        spec = AnchorSpec(
            name="freq",
            target_property="frequency",
            canonical_prompt="How often does [[s]] happen?",
            options=("never", "sometimes", "always"),
            is_ordered=True,
            semantic_pole_low="never",
            semantic_pole_high="always",
            required_span_labels=frozenset({"s"}),
        )
        anchor = spec.build()
        poles = anchor.response_space.semantic_poles
        assert poles is not None
        assert poles.as_tuple() == ("never", "always")

    def test_partial_poles_rejected(self) -> None:
        spec = AnchorSpec(
            name="x",
            target_property="x",
            canonical_prompt="Q?",
            options=("a", "b"),
            semantic_pole_low="a",
        )
        with pytest.raises(ValueError, match="only one pole"):
            spec.build()


class TestDriftConfig:
    """Tests for :class:`DriftConfig`."""

    def test_default_builds_structural_only(self) -> None:
        guard = DriftConfig().build()
        assert len(guard) == 1
        assert isinstance(guard.validators[0], StructuralDriftValidator)

    def test_embedding_requires_adapter(self) -> None:
        cfg = DriftConfig(enable_embedding=True)
        with pytest.raises(ValueError, match="embedding_adapter"):
            cfg.build()

    def test_perplexity_requires_adapter(self) -> None:
        cfg = DriftConfig(enable_perplexity=True)
        with pytest.raises(ValueError, match="perplexity_adapter"):
            cfg.build()

    def test_with_adapters(self) -> None:
        class Adapter:
            def get_embedding(self, text: str) -> Sequence[float]:
                del text
                return (1.0, 0.0)

            def compute_perplexity(self, text: str) -> float:
                del text
                return 25.0

        adapter = Adapter()
        cfg = DriftConfig(
            enable_embedding=True,
            enable_perplexity=True,
            max_perplexity=50.0,
        )
        guard = cfg.build(
            embedding_adapter=adapter,
            perplexity_adapter=adapter,
        )
        assert len(guard) == 3


class TestFamilySpec:
    """Tests for :class:`FamilySpec`."""

    def _anchor(self, name: str = "x") -> AnchorSpec:
        return AnchorSpec(
            name=name,
            target_property=name,
            canonical_prompt="Question for [[s]]?",
            options=("no", "yes"),
            is_ordered=False,
            required_span_labels=frozenset({"s"}),
        )

    def test_template_realization(self) -> None:
        spec = FamilySpec(
            anchor=self._anchor(),
            realization_kind="template",
            template="Did [[s]] happen?",
        )
        family = spec.build(
            drift_guard=DriftConfig().build(),
            lm_client=None,
            lm_model_name="",
            cache=None,
            lm_temperature=0.3,
            lm_max_tokens=200,
        )
        assert isinstance(family.realization, TemplateRealization)

    def test_contextual_requires_variants(self) -> None:
        spec = FamilySpec(
            anchor=self._anchor(),
            realization_kind="contextual",
            variants=(),
        )
        with pytest.raises(ValueError, match="variants is empty"):
            spec.build(
                drift_guard=DriftConfig().build(),
                lm_client=None,
                lm_model_name="",
                cache=None,
                lm_temperature=0.3,
                lm_max_tokens=200,
            )

    def test_contextual_realization(self) -> None:
        spec = FamilySpec(
            anchor=self._anchor(),
            realization_kind="contextual",
            variants=(
                TemplateVariantSpec(
                    template="V: [[s]]?",
                    condition_name="test_is_verb",
                    priority=10,
                ),
                TemplateVariantSpec(template="Generic [[s]]?", priority=0),
            ),
        )
        family = spec.build(
            drift_guard=DriftConfig().build(),
            lm_client=None,
            lm_model_name="",
            cache=None,
            lm_temperature=0.3,
            lm_max_tokens=200,
        )
        assert isinstance(family.realization, ContextualTemplateRealization)

    def test_lm_requires_client(self) -> None:
        spec = FamilySpec(
            anchor=self._anchor(),
            realization_kind="lm",
        )
        with pytest.raises(ValueError, match="no lm_client"):
            spec.build(
                drift_guard=DriftConfig().build(),
                lm_client=None,
                lm_model_name="x",
                cache=None,
                lm_temperature=0.3,
                lm_max_tokens=200,
            )

    def test_lm_realization(self) -> None:
        class Client:
            def complete(
                self, prompt: str, *, temperature: float, max_tokens: int
            ) -> str:
                del prompt, temperature, max_tokens
                return "Did [[s]] happen?"

        spec = FamilySpec(
            anchor=self._anchor(),
            realization_kind="lm",
        )
        family = spec.build(
            drift_guard=DriftConfig().build(),
            lm_client=Client(),
            lm_model_name="stub",
            cache=None,
            lm_temperature=0.5,
            lm_max_tokens=128,
        )
        assert isinstance(family.realization, LMRealization)

    def test_condition_name_resolved(self) -> None:
        spec = FamilySpec(
            anchor=self._anchor("y"),
            condition_name="test_is_verb",
            depends_on=(),
        )
        family = spec.build(
            drift_guard=DriftConfig().build(),
            lm_client=None,
            lm_model_name="",
            cache=None,
            lm_temperature=0.3,
            lm_max_tokens=200,
        )
        assert family.is_always_applicable is False
        assert family.is_applicable(ProtocolContext(target_upos="VERB"))
        assert not family.is_applicable(ProtocolContext(target_upos="NOUN"))


class TestProtocolConfig:
    """Tests for :class:`ProtocolConfig`."""

    def test_empty_default(self) -> None:
        cfg = ProtocolConfig()
        proto = cfg.build()
        assert isinstance(proto, AnnotationProtocol)
        assert len(proto) == 0

    def test_two_family_protocol(self) -> None:
        cfg = ProtocolConfig(
            name="aspect",
            families=(
                FamilySpec(
                    anchor=AnchorSpec(
                        name="change",
                        target_property="dynamicity",
                        canonical_prompt="Changing [[s]]?",
                        options=("no", "yes"),
                        is_ordered=False,
                        required_span_labels=frozenset({"s"}),
                    ),
                ),
                FamilySpec(
                    anchor=AnchorSpec(
                        name="completion",
                        target_property="telicity",
                        canonical_prompt="Endpoint [[s]]?",
                        options=("no", "yes"),
                        is_ordered=False,
                        required_span_labels=frozenset({"s"}),
                    ),
                    depends_on=("change",),
                ),
            ),
        )
        proto = cfg.build()
        assert proto.name == "aspect"
        assert [f.name for f in proto.families] == ["change", "completion"]
        assert proto.family_by_name("completion").depends_on == ("change",)

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        """Round-trip through YAML preserves the protocol structure."""
        del tmp_path  # only used for type assertion below
        cfg = ProtocolConfig(
            name="rt",
            families=(
                FamilySpec(
                    anchor=AnchorSpec(
                        name="q1",
                        target_property="q1",
                        canonical_prompt="[[s]]?",
                        options=("no", "yes"),
                        is_ordered=False,
                        required_span_labels=frozenset({"s"}),
                    ),
                ),
            ),
        )
        yaml_text = to_yaml(cfg, include_defaults=False)
        assert "rt" in yaml_text
        assert "q1" in yaml_text


def test_protocol_config_in_bead_config() -> None:
    """ProtocolConfig is wired into BeadConfig.protocol."""
    from bead.config import BeadConfig  # noqa: PLC0415

    config = BeadConfig()
    assert isinstance(config.protocol, ProtocolConfig)
    assert config.protocol.name == ""
    assert len(config.protocol.families) == 0
