"""Tests for morphological transforms."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bead.transforms.base import TransformContext, TransformRegistry
from bead.transforms.morphology import (
    GERUND,
    INFINITIVE,
    PAST_PARTICIPLE,
    PAST_TENSE,
    PRESENT_3SG,
    InflectionSpec,
    MorphologicalTransform,
    _is_infinitive,
    _is_past_participle,
    _is_past_tense,
    _is_present_3sg,
    _is_present_participle,
    register_morphological_transforms,
)


class TestFeaturePredicates:
    """Tests for individual feature-matching predicates."""

    def test_present_participle(self) -> None:
        assert _is_present_participle({"verb_form": "V.PTCP", "tense": "PRS"})

    def test_present_participle_rejects_past(self) -> None:
        assert not _is_present_participle({"verb_form": "V.PTCP", "tense": "PST"})

    def test_past_tense(self) -> None:
        assert _is_past_tense({"tense": "PST"})

    def test_past_tense_rejects_participle(self) -> None:
        assert not _is_past_tense({"tense": "PST", "verb_form": "V.PTCP"})

    def test_past_participle(self) -> None:
        assert _is_past_participle({"verb_form": "V.PTCP", "tense": "PST"})

    def test_present_3sg(self) -> None:
        assert _is_present_3sg({"tense": "PRS", "person": "3", "number": "SG"})

    def test_present_3sg_rejects_participle(self) -> None:
        assert not _is_present_3sg(
            {"tense": "PRS", "person": "3", "number": "SG", "verb_form": "V.PTCP"}
        )

    def test_infinitive(self) -> None:
        assert _is_infinitive({"pos": "V"})

    def test_infinitive_rejects_tensed(self) -> None:
        assert not _is_infinitive({"pos": "V", "tense": "PRS"})


class TestInflectionSpec:
    """Tests for InflectionSpec."""

    def test_fields(self) -> None:
        spec = InflectionSpec(
            name="test",
            predicate=lambda f: True,
            description="a test",
        )

        assert spec.name == "test"
        assert spec.description == "a test"
        assert spec.predicate({})

    def test_standard_specs_have_names(self) -> None:
        """All standard specs have non-empty names."""
        for spec in [GERUND, PAST_TENSE, PAST_PARTICIPLE, PRESENT_3SG, INFINITIVE]:
            assert spec.name
            assert spec.description


class TestMorphologicalTransform:
    """Tests for MorphologicalTransform with mocked UniMorph."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock UniMorphAdapter that returns known forms for 'run'."""
        adapter = MagicMock()

        def fake_fetch(query=None, language_code=None):
            forms = {
                "run": [
                    MagicMock(
                        form="running",
                        features={"pos": "V", "verb_form": "V.PTCP", "tense": "PRS"},
                    ),
                    MagicMock(
                        form="ran",
                        features={"pos": "V", "tense": "PST"},
                    ),
                    MagicMock(
                        form="run",
                        features={"pos": "V", "verb_form": "V.PTCP", "tense": "PST"},
                    ),
                    MagicMock(
                        form="runs",
                        features={
                            "pos": "V",
                            "tense": "PRS",
                            "person": "3",
                            "number": "SG",
                        },
                    ),
                    MagicMock(
                        form="run",
                        features={"pos": "V"},
                    ),
                ],
                "walk": [
                    MagicMock(
                        form="walking",
                        features={"pos": "V", "verb_form": "V.PTCP", "tense": "PRS"},
                    ),
                ],
            }
            return forms.get(query, [])

        adapter.fetch_items = fake_fetch
        return adapter

    def test_gerund_single_word(self, mock_adapter) -> None:
        """Gerund transform inflects a single-word span."""
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(lemma="run", tokens=["run"])
        result = t("run", ctx)

        assert result == "running"

    def test_gerund_multi_word_head_first(self, mock_adapter) -> None:
        """Gerund inflects only the head in a multi-word span."""
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(
            lemma="run",
            head_index=0,
            tokens=["run", "to", "the", "store"],
        )
        result = t("run to the store", ctx)

        assert result == "running to the store"

    def test_past_tense(self, mock_adapter) -> None:
        """Past tense transform."""
        t = MorphologicalTransform(PAST_TENSE, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(lemma="run", tokens=["run"])
        result = t("run", ctx)

        assert result == "ran"

    def test_present_3sg(self, mock_adapter) -> None:
        """Present 3sg transform."""
        t = MorphologicalTransform(PRESENT_3SG, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(lemma="run", tokens=["run"])
        result = t("run", ctx)

        assert result == "runs"

    def test_fallback_when_lemma_not_found(self, mock_adapter) -> None:
        """Returns original text when no inflection found."""
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(lemma="zzz", tokens=["zzz"])
        result = t("zzz", ctx)

        assert result == "zzz"

    def test_uses_head_token_as_lemma_fallback(self, mock_adapter) -> None:
        """When lemma is None, uses the head token for lookup."""
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(tokens=["walk"])
        result = t("walk", ctx)

        assert result == "walking"

    def test_head_index_non_zero(self, mock_adapter) -> None:
        """Non-zero head index inflects the correct token."""
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = mock_adapter

        ctx = TransformContext(
            lemma="run",
            head_index=2,
            tokens=["quickly", "go", "run"],
        )
        result = t("quickly go run", ctx)

        assert result == "quickly go running"

    def test_caching(self) -> None:
        """Repeated lookups for same lemma use cache."""
        adapter = MagicMock()
        adapter.fetch_items.return_value = [
            MagicMock(
                form="running",
                features={"pos": "V", "verb_form": "V.PTCP", "tense": "PRS"},
            ),
        ]
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = adapter

        ctx = TransformContext(lemma="run", tokens=["run"])
        t("run", ctx)
        t("run", ctx)

        # fetch_items should be called only once
        assert adapter.fetch_items.call_count == 1

    def test_empty_text_returns_original(self, mock_adapter) -> None:
        """Empty text is returned unchanged."""
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = mock_adapter

        assert t("", TransformContext()) == ""

    def test_adapter_error_falls_back(self) -> None:
        """If adapter raises, return original text."""
        adapter = MagicMock()
        adapter.fetch_items.side_effect = RuntimeError("network error")
        t = MorphologicalTransform(GERUND, language_code="eng")
        t._adapter = adapter

        ctx = TransformContext(lemma="run", tokens=["run"])
        result = t("run", ctx)

        assert result == "run"


class TestRegisterMorphologicalTransforms:
    """Tests for register_morphological_transforms."""

    def test_registers_all_standard_transforms(self) -> None:
        """All five standard transforms are registered."""
        reg = TransformRegistry()
        register_morphological_transforms(reg, "eng")

        expected = {
            "gerund",
            "past_tense",
            "past_participle",
            "present_3sg",
            "infinitive",
        }

        assert set(reg.available()) == expected

    def test_registered_transforms_are_morphological(self) -> None:
        """Each registered transform is a MorphologicalTransform."""
        reg = TransformRegistry()
        register_morphological_transforms(reg, "eng")

        for name in reg.available():
            t = reg.get(name)
            assert isinstance(t, MorphologicalTransform)
