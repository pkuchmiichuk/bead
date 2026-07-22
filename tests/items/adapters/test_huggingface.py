"""Tests for HuggingFace model adapters."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_mock import MockerFixture

from bead.items.adapters.huggingface import (
    HuggingFaceLanguageModel,
    HuggingFaceMaskedLanguageModel,
    HuggingFaceNLI,
)
from bead.items.cache import ModelOutputCache

# ============================================================================
# HuggingFaceLanguageModel Tests
# ============================================================================


def test_gpt2_initialization(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test GPT-2 language model initialization."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache, device="cpu")

    assert adapter.model_name == "gpt2"
    assert adapter.device == "cpu"
    assert adapter.cache is in_memory_cache


def test_gpt2_loads_with_requested_dtype(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test the requested dtype is passed through when loading the model."""
    load = mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel(
        "gpt2", in_memory_cache, device="cpu", dtype="bfloat16"
    )
    adapter._load_model()

    assert load.call_args.kwargs["dtype"] == "bfloat16"


def test_gpt2_defaults_to_checkpoint_dtype(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test the checkpoint's own dtype is used when none is requested."""
    load = mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache, device="cpu")
    adapter._load_model()

    assert load.call_args.kwargs["dtype"] == "auto"


def test_gpt2_compute_log_probability(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test log probability computation for causal LM."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache)

    log_prob = adapter.compute_log_probability(sample_texts[0])

    assert isinstance(log_prob, float)
    # Should be negative (mock loss is 2.5, 10 tokens -> -25.0)
    assert log_prob < 0


def test_gpt2_compute_perplexity(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test perplexity computation for causal LM."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache)

    perplexity = adapter.compute_perplexity(sample_texts[0])

    assert isinstance(perplexity, float)
    assert perplexity > 0  # Perplexity must be positive
    assert perplexity == pytest.approx(np.exp(2.5), rel=0.01)  # exp(mock loss)


def test_gpt2_get_embedding(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test embedding extraction for causal LM."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache)

    embedding = adapter.get_embedding(sample_texts[0])

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # Standard GPT-2 hidden size


def test_gpt2_compute_nli_not_supported(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that NLI is not supported for causal LMs."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache)

    with pytest.raises(NotImplementedError, match="NLI is not supported"):
        adapter.compute_nli("premise", "hypothesis")


def test_gpt2_caching(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that results are cached properly."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache)

    # First call - should compute
    text = "The cat sat on the mat."
    log_prob1 = adapter.compute_log_probability(text)

    # Second call - should hit cache
    log_prob2 = adapter.compute_log_probability(text)

    assert log_prob1 == log_prob2


def test_gpt2_device_fallback(
    mocker: MockerFixture,
    mock_gpt2_model: pytest.fixture,
    mock_gpt2_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test device fallback when CUDA unavailable."""
    mocker.patch(
        "bead.items.adapters.huggingface.torch.cuda.is_available",
        return_value=False,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForCausalLM.from_pretrained",
        return_value=mock_gpt2_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_gpt2_tokenizer,
    )

    adapter = HuggingFaceLanguageModel("gpt2", in_memory_cache, device="cuda")

    # Should fallback to CPU
    assert adapter.device == "cpu"


# ============================================================================
# HuggingFaceMaskedLanguageModel Tests
# ============================================================================


def test_bert_initialization(
    mocker: MockerFixture,
    mock_bert_model: pytest.fixture,
    mock_bert_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test BERT masked language model initialization."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForMaskedLM.from_pretrained",
        return_value=mock_bert_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_bert_tokenizer,
    )

    adapter = HuggingFaceMaskedLanguageModel(
        "bert-base-uncased", in_memory_cache, device="cpu"
    )

    assert adapter.model_name == "bert-base-uncased"
    assert adapter.device == "cpu"


def test_bert_compute_log_probability(
    mocker: MockerFixture,
    mock_bert_model: pytest.fixture,
    mock_bert_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test pseudo-log-likelihood computation for masked LM."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForMaskedLM.from_pretrained",
        return_value=mock_bert_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_bert_tokenizer,
    )

    adapter = HuggingFaceMaskedLanguageModel("bert-base-uncased", in_memory_cache)

    log_prob = adapter.compute_log_probability(sample_texts[0])

    assert isinstance(log_prob, float)
    # Pseudo-log-likelihood can be negative or positive


def test_bert_get_embedding(
    mocker: MockerFixture,
    mock_bert_model: pytest.fixture,
    mock_bert_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test [CLS] embedding extraction for masked LM."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForMaskedLM.from_pretrained",
        return_value=mock_bert_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_bert_tokenizer,
    )

    adapter = HuggingFaceMaskedLanguageModel("bert-base-uncased", in_memory_cache)

    embedding = adapter.get_embedding(sample_texts[0])

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # Standard BERT hidden size


def test_bert_compute_nli_not_supported(
    mocker: MockerFixture,
    mock_bert_model: pytest.fixture,
    mock_bert_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that NLI is not supported for masked LMs."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForMaskedLM.from_pretrained",
        return_value=mock_bert_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_bert_tokenizer,
    )

    adapter = HuggingFaceMaskedLanguageModel("bert-base-uncased", in_memory_cache)

    with pytest.raises(NotImplementedError, match="NLI is not supported"):
        adapter.compute_nli("premise", "hypothesis")


# ============================================================================
# HuggingFaceNLI Tests
# ============================================================================


def test_nli_initialization(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    mock_nli_config: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test NLI model initialization."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=mock_nli_config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache, device="cpu")

    assert adapter.model_name == "roberta-large-mnli"
    assert adapter.device == "cpu"


def test_nli_compute_nli(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    mock_nli_config: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    expected_nli_scores: dict[str, float],
) -> None:
    """Test NLI score computation."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=mock_nli_config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache)

    scores = adapter.compute_nli(
        premise="Mary loves reading books.", hypothesis="Mary enjoys literature."
    )

    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"entailment", "neutral", "contradiction"}
    # Scores should sum to ~1.0
    assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)
    # Check values match expected
    for key in scores:
        assert scores[key] == pytest.approx(expected_nli_scores[key], abs=0.01)


def test_nli_get_embedding(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    mock_nli_config: pytest.fixture,
    in_memory_cache: ModelOutputCache,
    sample_texts: list[str],
) -> None:
    """Test embedding extraction from NLI model."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=mock_nli_config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache)

    embedding = adapter.get_embedding(sample_texts[0])

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)  # Standard RoBERTa hidden size


def test_nli_log_probability_not_supported(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    mock_nli_config: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that log probability is not supported for NLI models."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=mock_nli_config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache)

    with pytest.raises(NotImplementedError, match="Log probability is not supported"):
        adapter.compute_log_probability("text")


def test_nli_perplexity_not_supported(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    mock_nli_config: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that perplexity is not supported for NLI models."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=mock_nli_config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache)

    with pytest.raises(NotImplementedError, match="Perplexity is not supported"):
        adapter.compute_perplexity("text")


def test_nli_label_mapping(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test label mapping from config."""
    # Create config with uppercase labels
    config = mocker.Mock()
    config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache)

    scores = adapter.compute_nli("premise", "hypothesis")

    # Should normalize labels to lowercase
    assert "entailment" in scores
    assert "neutral" in scores
    assert "contradiction" in scores


def test_nli_caching(
    mocker: MockerFixture,
    mock_nli_model: pytest.fixture,
    mock_nli_tokenizer: pytest.fixture,
    mock_nli_config: pytest.fixture,
    in_memory_cache: ModelOutputCache,
) -> None:
    """Test that NLI results are cached properly."""
    mocker.patch(
        "bead.items.adapters.huggingface.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_nli_model,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoTokenizer.from_pretrained",
        return_value=mock_nli_tokenizer,
    )
    mocker.patch(
        "bead.items.adapters.huggingface.AutoConfig.from_pretrained",
        return_value=mock_nli_config,
    )

    adapter = HuggingFaceNLI("roberta-large-mnli", in_memory_cache)

    # First call - should compute
    scores1 = adapter.compute_nli("Mary loves books.", "Mary enjoys reading.")

    # Second call - should hit cache
    scores2 = adapter.compute_nli("Mary loves books.", "Mary enjoys reading.")

    assert scores1 == scores2
