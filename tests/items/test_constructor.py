"""Tests for ItemConstructor."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import numpy as np
import pytest

from bead.dsl.context import EvaluationContext
from bead.dsl.parser import parse
from bead.items.adapters.base import ModelAdapter
from bead.items.adapters.registry import ModelAdapterRegistry
from bead.items.cache import ModelOutputCache
from bead.items.constructor import ItemConstructor
from bead.items.item_template import (
    ItemElement,
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.templates.filler import FilledTemplate


@pytest.fixture
def cache(tmp_path):
    """Create test cache."""
    return ModelOutputCache(cache_dir=tmp_path / "cache", backend="filesystem")


@pytest.fixture
def mock_adapter_class():
    """Create mock model adapter class."""

    class MockAdapter(ModelAdapter):
        def __init__(self, model_name: str, cache: ModelOutputCache, **kwargs) -> None:
            super().__init__(model_name, cache, model_version="1.0")

        def compute_log_probability(self, text: str) -> float:
            # Return different values for different text for testing
            return -len(text) * 2.5

        def compute_perplexity(self, text: str) -> float:
            return len(text) * 0.5 + 10.0

        def get_embedding(self, text: str):
            return np.array([0.1 * i for i in range(384)])

        def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
            # Return high entailment if texts are similar length
            if abs(len(premise) - len(hypothesis)) < 5:
                return {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05}
            return {"entailment": 0.1, "neutral": 0.3, "contradiction": 0.6}

        def compute_similarity(self, text1: str, text2: str) -> float:
            # Simple similarity based on length difference
            return max(0.0, 1.0 - abs(len(text1) - len(text2)) / 100.0)

    return MockAdapter


@pytest.fixture
def registry(mock_adapter_class):
    """Create test registry with mock adapter."""
    reg = ModelAdapterRegistry()
    reg.register("huggingface_lm", mock_adapter_class)
    reg.register("huggingface_nli", mock_adapter_class)
    reg.register("sentence_transformer", mock_adapter_class)
    return reg


@pytest.fixture
def constructor(registry, cache):
    """Create test constructor."""
    return ItemConstructor(registry, cache)


class TestItemConstructor:
    """Tests for ItemConstructor class."""

    def test_init(self, registry, cache) -> None:
        """Test constructor initialization."""
        constructor = ItemConstructor(registry, cache)
        assert constructor.model_registry is registry
        assert constructor.cache is cache
        assert constructor.constraint_resolver is None

    def test_render_elements_text_only(self, constructor) -> None:
        """Test rendering text-only elements."""
        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="sentence",
                    content="The cat sat.",
                )
            ],
        )

        rendered = constructor._render_elements(template, {})

        assert rendered == {"sentence": "The cat sat."}

    def test_render_elements_with_template_ref(self, constructor) -> None:
        """Test rendering elements with filled template references."""
        ref_id = uuid4()
        filled_template = FilledTemplate(
            template_id="t1",
            template_name="transitive",
            slot_fillers={
                "subject": LexicalItem(
                    lemma="cat", language_code="eng", features={"pos": "NOUN"}
                ),
                "verb": LexicalItem(
                    lemma="broke", language_code="eng", features={"pos": "VERB"}
                ),
            },
            rendered_text="The cat broke the vase",
            strategy_name="exhaustive",
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Natural?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="filled_template_ref",
                    element_name="sentence",
                    filled_template_ref_id=ref_id,
                )
            ],
        )

        rendered = constructor._render_elements(template, {ref_id: filled_template})

        assert rendered == {"sentence": "The cat broke the vase"}

    def test_render_elements_missing_ref(self, constructor) -> None:
        """Test error when filled template reference is missing."""
        missing_id = uuid4()
        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="filled_template_ref",
                    element_name="sentence",
                    filled_template_ref_id=missing_id,
                )
            ],
        )

        with pytest.raises(
            (ValueError, dx.ValidationError), match="references missing"
        ):
            constructor._render_elements(template, {})

    def test_render_elements_multiple(self, constructor) -> None:
        """Test rendering multiple elements."""
        ref_id = uuid4()
        filled = FilledTemplate(
            template_id="t1",
            template_name="test",
            slot_fillers={},
            rendered_text="Filled text",
            strategy_name="exhaustive",
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="context",
                    content="Context text",
                ),
                ItemElement(
                    element_type="filled_template_ref",
                    element_name="target",
                    filled_template_ref_id=ref_id,
                ),
            ],
        )

        rendered = constructor._render_elements(template, {ref_id: filled})

        assert rendered == {"context": "Context text", "target": "Filled text"}

    def test_construct_items_no_constraints(self, constructor) -> None:
        """Test constructing items without constraints."""
        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Natural?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="sentence",
                    content="Test sentence",
                )
            ],
            constraints=[],  # No constraints
        )

        items = list(constructor.construct_items(template, {}, {}))

        assert len(items) == 1
        item = items[0]
        assert item.item_template_id == template.id
        assert item.rendered_elements == {"sentence": "Test sentence"}
        assert item.constraint_satisfaction == ()
        assert item.model_outputs == ()

    def test_construct_items_with_dsl_constraint(self, constructor) -> None:
        """Test constructing items with DSL constraint."""
        constraint_id = uuid4()
        constraint = Constraint(
            expression="len(sentence) > 5",
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Natural?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="sentence",
                    content="Test sentence",
                )
            ],
            constraints=[constraint_id],
        )

        items = list(
            constructor.construct_items(template, {}, {constraint_id: constraint})
        )

        assert len(items) == 1
        item = items[0]
        by_id = {cs.constraint_id: cs.satisfied for cs in item.constraint_satisfaction}
        assert constraint_id in by_id
        assert by_id[constraint_id] is True

    def test_construct_items_constraint_not_satisfied(self, constructor) -> None:
        """Test that items failing constraints are not yielded."""
        constraint_id = uuid4()
        constraint = Constraint(
            expression="len(sentence) > 100",  # Will fail for short text
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Natural?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text", element_name="sentence", content="Short"
                )
            ],
            constraints=[constraint_id],
        )

        items = list(
            constructor.construct_items(template, {}, {constraint_id: constraint})
        )

        # Item should not be yielded as constraint fails
        assert len(items) == 0

    def test_construct_items_with_model_constraint(self, constructor) -> None:
        """Test constructing items with model-based DSL constraint."""
        constraint_id = uuid4()
        # Constraint using lm_prob function
        constraint = Constraint(
            expression='lm_prob(sentence, "gpt2") > -100',
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Natural?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="sentence",
                    content="The cat sat on the mat",
                )
            ],
            constraints=[constraint_id],
        )

        items = list(
            constructor.construct_items(template, {}, {constraint_id: constraint})
        )

        assert len(items) == 1
        item = items[0]
        # Check that model outputs were created
        assert len(item.model_outputs) > 0
        # Check log probability was computed
        assert any(
            output.operation == "log_probability" for output in item.model_outputs
        )

    def test_extract_model_calls_lm_prob(self, constructor) -> None:
        """Test extracting lm_prob function calls from AST."""
        ast = parse('lm_prob(sentence, "gpt2")')
        calls = constructor._extract_model_calls(ast, {"sentence": "Test text"})

        assert len(calls) == 1
        assert calls[0]["function"] == "lm_prob"
        assert calls[0]["text"] == "Test text"
        assert calls[0]["model"] == "gpt2"
        assert calls[0]["operation"] == "log_probability"

    def test_extract_model_calls_nli(self, constructor) -> None:
        """Test extracting NLI function calls from AST."""
        ast = parse('nli(premise, hypothesis, "roberta-nli")')
        calls = constructor._extract_model_calls(
            ast, {"premise": "P text", "hypothesis": "H text"}
        )

        assert len(calls) == 1
        assert calls[0]["function"] == "nli"
        assert calls[0]["premise"] == "P text"
        assert calls[0]["hypothesis"] == "H text"
        assert calls[0]["model"] == "roberta-nli"
        assert calls[0]["operation"] == "nli"

    def test_extract_model_calls_complex_expression(self, constructor) -> None:
        """Test extracting calls from complex DSL expression."""
        # Expression with multiple model calls
        ast = parse('lm_prob(sent, "gpt2") > -50 and len(sent) > 10')
        calls = constructor._extract_model_calls(ast, {"sent": "Test sentence here"})

        assert len(calls) == 1
        assert calls[0]["function"] == "lm_prob"

    def test_execute_model_call_log_probability(self, constructor) -> None:
        """Test executing log probability model call."""
        call_spec = {
            "function": "lm_prob",
            "text": "Test sentence",
            "model": "gpt2",
            "operation": "log_probability",
        }

        output = constructor._execute_model_call(call_spec)

        assert output is not None
        assert output.operation == "log_probability"
        assert output.model_name == "gpt2"
        assert isinstance(output.output, float)

    def test_execute_model_call_nli(self, constructor) -> None:
        """Test executing NLI model call."""
        call_spec = {
            "function": "nli",
            "premise": "The cat is sleeping",
            "hypothesis": "The cat is resting",
            "model": "roberta-nli",
            "operation": "nli",
        }

        output = constructor._execute_model_call(call_spec)

        assert output is not None
        assert output.operation == "nli"
        assert isinstance(output.output, dict)
        assert "entailment" in output.output
        assert "neutral" in output.output
        assert "contradiction" in output.output

    def test_execute_model_call_caching(self, constructor, cache) -> None:
        """Test that model calls can be cached."""
        call_spec = {
            "function": "lm_prob",
            "text": "Cached text",
            "model": "gpt2",
            "operation": "log_probability",
        }

        # First call
        output1 = constructor._execute_model_call(call_spec)
        assert output1 is not None
        assert isinstance(output1.output, float)

        # Pre-populate cache manually for second call
        cache.set("gpt2", "log_probability", output1.output, text="Cached text")

        # Second call should use pre-populated cache
        output2 = constructor._execute_model_call(call_spec)
        assert output2 is not None
        assert output2.computation_metadata.get("from_cache") is True
        assert output1.output == output2.output

    def test_check_constraints_missing_constraint(self, constructor) -> None:
        """Test error when constraint UUID not found."""
        constraint_id = uuid4()
        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            constraints=[constraint_id],
        )

        with pytest.raises((ValueError, dx.ValidationError), match="not found"):
            constructor._check_constraints(template, {}, [], {})

    def test_compute_model_outputs_missing_constraint(self, constructor) -> None:
        """Test error when computing outputs for missing constraint."""
        constraint_id = uuid4()
        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            constraints=[constraint_id],
        )

        with pytest.raises((ValueError, dx.ValidationError), match="not found"):
            constructor._compute_model_outputs(template, {}, {})

    def test_construct_items_preserves_filled_refs(self, constructor) -> None:
        """Test that constructed items preserve filled template references."""
        ref_id = uuid4()
        filled = FilledTemplate(
            template_id="t1",
            template_name="test",
            slot_fillers={},
            rendered_text="Text",
            strategy_name="exhaustive",
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="filled_template_ref",
                    element_name="sent",
                    filled_template_ref_id=ref_id,
                )
            ],
        )

        items = list(constructor.construct_items(template, {ref_id: filled}, {}))

        assert len(items) == 1
        assert items[0].filled_template_refs == (ref_id,)

    def test_construct_items_yields_iterator(self, constructor) -> None:
        """Test that construct_items returns an iterator."""
        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Test?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(element_type="text", element_name="s", content="text")
            ],
        )

        result = constructor.construct_items(template, {}, {})

        # Check it's an iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

        # Consume it
        items = list(result)
        assert len(items) == 1


class TestModelFunctionRegistration:
    """Tests for model function registration in DSL context."""

    def test_register_model_functions(self, constructor) -> None:
        """Test that model functions are registered in context."""
        context = EvaluationContext()
        constructor._register_model_functions(context, [])

        # Check that functions are registered
        assert "lm_prob" in context._functions
        assert "lm_perplexity" in context._functions
        assert "nli" in context._functions
        assert "similarity" in context._functions
        assert "embedding" in context._functions

    def test_lm_prob_function(self, constructor, cache) -> None:
        """Test lm_prob function uses cache."""
        # Pre-populate cache
        cache.set("gpt2", "log_probability", -42.0, text="test")

        context = EvaluationContext()
        constructor._register_model_functions(context, [])

        # Call function
        result = context.call_function("lm_prob", ["test", "gpt2"])

        assert result == -42.0

    def test_nli_function(self, constructor, cache) -> None:
        """Test nli function uses cache."""
        # Pre-populate cache
        nli_scores = {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02}
        cache.set("roberta-nli", "nli", nli_scores, premise="p", hypothesis="h")

        context = EvaluationContext()
        constructor._register_model_functions(context, [])

        # Call function (note: DSL passes as positional args)
        result = context.call_function("nli", ["p", "h", "roberta-nli"])

        assert result == nli_scores

    def test_similarity_function(self, constructor, cache) -> None:
        """Test similarity function uses cache."""
        # Pre-populate cache
        cache.set("model", "similarity", 0.85, text1="a", text2="b")

        context = EvaluationContext()
        constructor._register_model_functions(context, [])

        result = context.call_function("similarity", ["a", "b", "model"])

        assert result == 0.85

    def test_embedding_function(self, constructor, cache) -> None:
        """Test embedding function uses cache."""
        # Pre-populate cache with numpy array
        emb = np.array([0.1, 0.2, 0.3])
        cache.set("model", "embedding", emb, text="test")

        context = EvaluationContext()
        constructor._register_model_functions(context, [])

        result = context.call_function("embedding", ["test", "model"])

        assert result == [0.1, 0.2, 0.3]


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow_with_model_constraints(self, constructor) -> None:
        """Test complete item construction with model-based constraints."""
        # Create template with model constraint
        constraint_id = uuid4()
        constraint = Constraint(
            expression='lm_prob(sentence, "gpt2") > -100',
        )

        template = ItemTemplate(
            name="acceptability_test",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural?", scale_bounds=ScaleBounds(min=1, max=7)
            ),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="sentence",
                    content="The cat sat on the mat",
                )
            ],
            constraints=[constraint_id],
        )

        # Construct items
        items = list(
            constructor.construct_items(template, {}, {constraint_id: constraint})
        )

        # Verify
        assert len(items) == 1
        item = items[0]
        assert item.rendered_elements["sentence"] == "The cat sat on the mat"
        assert len(item.model_outputs) > 0
        assert {cs.constraint_id: cs.satisfied for cs in item.constraint_satisfaction}[
            constraint_id
        ] is True

    def test_multiple_constraints(self, constructor) -> None:
        """Test item construction with multiple constraints."""
        c1_id, c2_id = uuid4(), uuid4()
        c1 = Constraint(expression="len(sentence) > 5")
        c2 = Constraint(expression="len(sentence) < 100")

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Natural?"),
            presentation_spec=PresentationSpec(mode="static"),
            elements=[
                ItemElement(
                    element_type="text",
                    element_name="sentence",
                    content="Test sentence",
                )
            ],
            constraints=[c1_id, c2_id],
        )

        items = list(constructor.construct_items(template, {}, {c1_id: c1, c2_id: c2}))

        assert len(items) == 1
        cs_map = {
            cs.constraint_id: cs.satisfied for cs in items[0].constraint_satisfaction
        }
        assert cs_map[c1_id] is True
        assert cs_map[c2_id] is True
