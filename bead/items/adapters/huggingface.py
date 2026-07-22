"""HuggingFace model adapters for language models and NLI.

This module provides adapters for HuggingFace Transformers models:
- HuggingFaceLanguageModel: Causal LMs (GPT-2, GPT-Neo, Llama, Mistral)
- HuggingFaceMaskedLanguageModel: Masked LMs (BERT, RoBERTa, ALBERT)
- HuggingFaceNLI: NLI models (RoBERTa-MNLI, DeBERTa-MNLI, BART-MNLI)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import psutil
import torch
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bead.adapters.huggingface import DeviceType, HuggingFaceAdapterMixin
from bead.items.adapters.base import ModelAdapter
from bead.items.cache import ModelOutputCache

if TYPE_CHECKING:
    from transformers.models.auto.configuration_auto import AutoConfig

logger = logging.getLogger(__name__)


class HuggingFaceLanguageModel(HuggingFaceAdapterMixin, ModelAdapter):
    """Adapter for HuggingFace causal language models.

    Supports models like GPT-2, GPT-Neo, Llama, Mistral, and other
    autoregressive (left-to-right) language models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "gpt2", "gpt2-medium").
    cache : ModelOutputCache
        Cache instance for storing model outputs.
    device : {"cpu", "cuda", "mps"}
        Device to run model on. Falls back to CPU if device unavailable.
    model_version : str
        Version string for cache tracking.
    dtype : str
        Torch dtype to load the weights in, such as ``"bfloat16"``. Defaults to
        ``"auto"``, which keeps the dtype the checkpoint was saved in; loading a
        half-precision checkpoint as float32 would double its memory.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.items.cache import ModelOutputCache
    >>> cache = ModelOutputCache(cache_dir=Path(".cache"))
    >>> model = HuggingFaceLanguageModel("gpt2", cache, device="cpu")
    >>> log_prob = model.compute_log_probability("The cat sat on the mat.")
    >>> perplexity = model.compute_perplexity("The cat sat on the mat.")
    >>> embedding = model.get_embedding("The cat sat on the mat.")
    """

    def __init__(
        self,
        model_name: str,
        cache: ModelOutputCache,
        device: DeviceType = "cpu",
        model_version: str = "unknown",
        dtype: str = "auto",
    ) -> None:
        super().__init__(model_name, cache, model_version)
        self.device = self._validate_device(device)
        self.dtype = dtype
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def _load_model(self) -> None:
        """Load model and tokenizer lazily on first use."""
        if self._model is None:
            logger.info(f"Loading causal LM: {self.model_name}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=self.dtype
            )
            self._model.to(self.device)
            self._model.eval()

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # set padding token for models that don't have one
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def model(self) -> PreTrainedModel:
        """Get the model, loading if necessary."""
        self._load_model()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer, loading if necessary."""
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text under language model.

        Uses the model's loss with labels=input_ids to compute the negative
        log-likelihood of the text.

        Parameters
        ----------
        text : str
            Text to compute log probability for.

        Returns
        -------
        float
            Log probability of the text.
        """
        # Check cache
        cached = self.cache.get(self.model_name, "log_probability", text=text)
        if cached is not None:
            return cached

        # tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # compute loss (negative log-likelihood)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss.item()

        # loss is negative log-likelihood per token, convert to total log prob
        log_prob = -loss * input_ids.size(1)

        # cache result
        self.cache.set(
            self.model_name,
            "log_probability",
            log_prob,
            model_version=self.model_version,
            text=text,
        )

        return log_prob

    def _infer_optimal_batch_size(self) -> int:
        """Infer optimal batch size based on available resources.

        Considers:
        - Device type (CPU, CUDA, MPS)
        - Available memory
        - Model size
        - Sequence length estimates

        Returns
        -------
        int
            Recommended batch size.
        """
        # estimate model size
        model_params = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )

        if self.device == "cuda":
            try:
                # get GPU memory
                free_memory, _ = torch.cuda.mem_get_info(self.device)

                # conservative estimate: allow model + 4x model size for activations
                # reserve 20% for safety margin
                available_for_batch = (free_memory * 0.8) - model_params
                memory_per_item = model_params * 4  # very rough estimate

                batch_size = int(available_for_batch / memory_per_item)

                # clamp between reasonable bounds
                batch_size = max(8, min(batch_size, 256))

                free_gb = free_memory / 1e9
                model_gb = model_params / 1e9
                logger.info(
                    f"Inferred batch size {batch_size} for CUDA "
                    f"(free: {free_gb:.1f}GB, model: {model_gb:.2f}GB)"
                )
                return batch_size

            except Exception as e:
                logger.warning(
                    f"Failed to infer CUDA batch size: {e}, using default 32"
                )
                return 32

        elif self.device == "mps":
            try:
                # mps (Apple Silicon) - use system RAM as proxy
                # mps shares unified memory with system
                available_memory = psutil.virtual_memory().available

                # reserve 4GB for system + model
                available_for_batch = max(
                    0, available_memory - (4 * 1024**3) - model_params
                )
                memory_per_item = model_params * 3  # mps is more efficient than CUDA

                batch_size = int(available_for_batch / memory_per_item)

                # clamp between reasonable bounds
                batch_size = max(8, min(batch_size, 256))

                avail_gb = available_memory / 1e9
                model_gb = model_params / 1e9
                logger.info(
                    f"Inferred batch size {batch_size} for MPS "
                    f"(available: {avail_gb:.1f}GB, model: {model_gb:.2f}GB)"
                )
                return batch_size

            except Exception as e:
                logger.warning(f"Failed to infer MPS batch size: {e}, using default 64")
                return 64

        else:  # CPU
            try:
                # cpu - check available RAM
                available_memory = psutil.virtual_memory().available

                # reserve 2GB for system + model
                available_for_batch = max(
                    0, available_memory - (2 * 1024**3) - model_params
                )
                memory_per_item = model_params * 2  # cpu has less overhead than GPU

                batch_size = int(available_for_batch / memory_per_item)

                # clamp between reasonable bounds
                batch_size = max(4, min(batch_size, 128))

                avail_gb = available_memory / 1e9
                model_gb = model_params / 1e9
                logger.info(
                    f"Inferred batch size {batch_size} for CPU "
                    f"(available: {avail_gb:.1f}GB, model: {model_gb:.2f}GB)"
                )
                return batch_size

            except Exception as e:
                logger.warning(f"Failed to infer CPU batch size: {e}, using default 16")
                return 16

    def compute_log_probability_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[float]:
        """Compute log probabilities for multiple texts efficiently.

        Uses batched tokenization and inference for significant speedup.
        Checks cache before computing, only processes uncached texts.

        Parameters
        ----------
        texts : list[str]
            Texts to compute log probabilities for.
        batch_size : int | None, default=None
            Number of texts to process in each batch. If None, automatically
            infers optimal batch size based on available device memory and
            model size.

        Returns
        -------
        list[float]
            Log probabilities for each text, in the same order as input.

        Examples
        --------
        >>> texts = ["The cat sat.", "The dog ran.", "The bird flew."]
        >>> log_probs = model.compute_log_probability_batch(texts)
        >>> len(log_probs) == len(texts)
        True
        """
        # infer batch size if not provided
        if batch_size is None:
            batch_size = self._infer_optimal_batch_size()

        # check cache for all texts
        results: list[float | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(self.model_name, "log_probability", text=text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # if everything was cached, return immediately
        if not uncached_texts:
            logger.info(f"All {len(texts)} texts found in cache")
            return [r for r in results if r is not None]

        # log cache statistics
        n_cached = len(texts) - len(uncached_texts)
        cache_rate = (n_cached / len(texts)) * 100 if texts else 0
        logger.info(
            f"Cache: {n_cached}/{len(texts)} texts ({cache_rate:.1f}%), "
            f"processing {len(uncached_texts)} uncached with batch_size={batch_size}"
        )

        # process uncached texts in batches with progress tracking
        uncached_scores: list[float] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Scoring with {self.model_name}[/cyan]",
                total=len(uncached_texts),
            )

            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[batch_start : batch_start + batch_size]
                batch_scores = self._process_batch(batch_texts)
                uncached_scores.extend(batch_scores)
                progress.update(task, advance=len(batch_texts))

        # merge cached and newly computed results
        uncached_iter = iter(uncached_scores)
        final_results: list[float] = []
        for result in results:
            if result is None:
                final_results.append(next(uncached_iter))
            else:
                final_results.append(result)

        return final_results

    def _process_batch(self, batch_texts: list[str]) -> list[float]:
        """Process a single batch of texts and return scores.

        Parameters
        ----------
        batch_texts : list[str]
            Texts to process in this batch.

        Returns
        -------
        list[float]
            Log probabilities for each text.
        """
        batch_scores: list[float] = []

        # tokenize batch
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # compute losses for batch
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            # for batched inputs, we need to compute loss per item
            # the model returns average loss across batch, so we need
            # to compute per-item losses manually
            logits = outputs.logits  # [batch, seq_len, vocab]

            # shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention = attention_mask[..., 1:].contiguous()

            # compute log probabilities per token
            log_probs_per_token = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            # gather log probs for actual tokens
            gathered_log_probs = torch.gather(
                log_probs_per_token,
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)

            # mask padding tokens and sum per sequence
            masked_log_probs = gathered_log_probs * shift_attention
            sequence_log_probs = masked_log_probs.sum(dim=1)

        # convert to list and cache
        for text, log_prob_tensor in zip(batch_texts, sequence_log_probs, strict=True):
            log_prob = log_prob_tensor.item()
            batch_scores.append(log_prob)

            # cache result
            self.cache.set(
                self.model_name,
                "log_probability",
                log_prob,
                model_version=self.model_version,
                text=text,
            )

        return batch_scores

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Perplexity is exp(average negative log-likelihood per token).

        Parameters
        ----------
        text : str
            Text to compute perplexity for.

        Returns
        -------
        float
            Perplexity of the text (positive value).
        """
        # check cache
        cached = self.cache.get(self.model_name, "perplexity", text=text)
        if cached is not None:
            return cached

        # tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # compute loss
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss.item()

        # perplexity is exp(loss)
        perplexity = np.exp(loss)

        # cache result
        self.cache.set(
            self.model_name,
            "perplexity",
            perplexity,
            model_version=self.model_version,
            text=text,
        )

        return float(perplexity)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Uses mean pooling of last hidden states as the text embedding.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.
        """
        # check cache
        cached = self.cache.get(self.model_name, "embedding", text=text)
        if cached is not None:
            return cached

        # tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]  # last layer

        # mean pooling (weighted by attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embedding = (sum_hidden / sum_mask).squeeze(0).cpu().numpy()

        # cache result
        self.cache.set(
            self.model_name,
            "embedding",
            embedding,
            model_version=self.model_version,
            text=text,
        )

        return embedding

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores.

        Not supported for causal language models.

        Raises
        ------
        NotImplementedError
            Always raised, as causal LMs don't support NLI directly.
        """
        raise NotImplementedError(
            f"NLI is not supported for causal language model {self.model_name}. "
            "Use HuggingFaceNLI adapter with an NLI-trained model instead."
        )


class HuggingFaceMaskedLanguageModel(HuggingFaceAdapterMixin, ModelAdapter):
    """Adapter for HuggingFace masked language models.

    Supports models like BERT, RoBERTa, ALBERT, and other masked language
    models (MLMs).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "bert-base-uncased").
    cache : ModelOutputCache
        Cache instance for storing model outputs.
    device : {"cpu", "cuda", "mps"}
        Device to run model on. Falls back to CPU if device unavailable.
    model_version : str
        Version string for cache tracking.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.items.cache import ModelOutputCache
    >>> cache = ModelOutputCache(cache_dir=Path(".cache"))
    >>> model = HuggingFaceMaskedLanguageModel("bert-base-uncased", cache)
    >>> log_prob = model.compute_log_probability("The cat sat on the mat.")
    >>> embedding = model.get_embedding("The cat sat on the mat.")
    """

    def __init__(
        self,
        model_name: str,
        cache: ModelOutputCache,
        device: DeviceType = "cpu",
        model_version: str = "unknown",
    ) -> None:
        super().__init__(model_name, cache, model_version)
        self.device = self._validate_device(device)
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def _load_model(self) -> None:
        """Load model and tokenizer lazily on first use."""
        if self._model is None:
            logger.info(f"Loading masked LM: {self.model_name}")
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def model(self) -> PreTrainedModel:
        """Get the model, loading if necessary."""
        self._load_model()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer, loading if necessary."""
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text using pseudo-log-likelihood.

        For MLMs, we use pseudo-log-likelihood: mask each token one at a time
        and sum the log probabilities of predicting each token.

        This is computationally expensive - caching is critical.

        Parameters
        ----------
        text : str
            Text to compute log probability for.

        Returns
        -------
        float
            Pseudo-log-probability of the text.
        """
        # check cache
        cached = self.cache.get(self.model_name, "log_probability", text=text)
        if cached is not None:
            return cached

        # tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        # compute pseudo-log-likelihood by masking each token
        total_log_prob = 0.0
        num_tokens = input_ids.size(1)

        with torch.no_grad():
            for i in range(num_tokens):
                # skip special tokens
                if input_ids[0, i] in [
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    continue

                # create masked version
                masked_input = input_ids.clone()
                original_token = masked_input[0, i].item()
                masked_input[0, i] = self.tokenizer.mask_token_id

                # get prediction
                outputs = self.model(masked_input)
                logits = outputs.logits[0, i]  # logits for masked position

                # compute log probability of original token
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[original_token].item()

        # cache result
        self.cache.set(
            self.model_name,
            "log_probability",
            total_log_prob,
            model_version=self.model_version,
            text=text,
        )

        return total_log_prob

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity based on pseudo-log-likelihood.

        Parameters
        ----------
        text : str
            Text to compute perplexity for.

        Returns
        -------
        float
            Perplexity of the text (positive value).
        """
        # check cache
        cached = self.cache.get(self.model_name, "perplexity", text=text)
        if cached is not None:
            return cached

        # get log probability
        log_prob = self.compute_log_probability(text)

        # count non-special tokens
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        num_tokens = sum(
            1
            for token_id in input_ids[0].tolist()
            if token_id
            not in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]
        )

        # perplexity is exp(-log_prob / num_tokens)
        perplexity = np.exp(-log_prob / max(num_tokens, 1))

        # cache result
        self.cache.set(
            self.model_name,
            "perplexity",
            perplexity,
            model_version=self.model_version,
            text=text,
        )

        return float(perplexity)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Uses the [CLS] token embedding from the last layer.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.
        """
        # check cache
        cached = self.cache.get(self.model_name, "embedding", text=text)
        if cached is not None:
            return cached

        # tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # use [CLS] token from last layer
            hidden_states = outputs.hidden_states[-1]
            cls_embedding = hidden_states[0, 0].cpu().numpy()

        # cache result
        self.cache.set(
            self.model_name,
            "embedding",
            cls_embedding,
            model_version=self.model_version,
            text=text,
        )

        return cls_embedding

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores.

        Not supported for masked language models.

        Raises
        ------
        NotImplementedError
            Always raised, as MLMs don't support NLI directly.
        """
        raise NotImplementedError(
            f"NLI is not supported for masked language model {self.model_name}. "
            "Use HuggingFaceNLI adapter with an NLI-trained model instead."
        )


class HuggingFaceNLI(HuggingFaceAdapterMixin, ModelAdapter):
    """Adapter for HuggingFace NLI models.

    Supports NLI models trained on MNLI and similar datasets
    (e.g., "roberta-large-mnli", "microsoft/deberta-base-mnli").

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for NLI model.
    cache : ModelOutputCache
        Cache instance for storing model outputs.
    device : {"cpu", "cuda", "mps"}
        Device to run model on. Falls back to CPU if device unavailable.
    model_version : str
        Version string for cache tracking.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.items.cache import ModelOutputCache
    >>> cache = ModelOutputCache(cache_dir=Path(".cache"))
    >>> nli = HuggingFaceNLI("roberta-large-mnli", cache, device="cpu")
    >>> scores = nli.compute_nli(
    ...     premise="Mary loves reading books.",
    ...     hypothesis="Mary enjoys literature."
    ... )
    >>> label = nli.get_nli_label(
    ...     premise="Mary loves reading books.",
    ...     hypothesis="Mary enjoys literature."
    ... )
    """

    def __init__(
        self,
        model_name: str,
        cache: ModelOutputCache,
        device: DeviceType = "cpu",
        model_version: str = "unknown",
    ) -> None:
        super().__init__(model_name, cache, model_version)
        self.device = self._validate_device(device)
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._label_mapping: dict[str, str] = {}

    def _load_model(self) -> None:
        """Load model and tokenizer lazily on first use."""
        if self._model is None:
            logger.info(f"Loading NLI model: {self.model_name}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()

            # Get label mapping from config
            config = AutoConfig.from_pretrained(self.model_name)
            if hasattr(config, "id2label"):
                # Build mapping from model labels to standard labels
                self._label_mapping = self._build_label_mapping(config.id2label)
            else:
                # Default mapping (assume standard order)
                self._label_mapping = {
                    "0": "entailment",
                    "1": "neutral",
                    "2": "contradiction",
                }

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _build_label_mapping(self, id2label: dict[int, str]) -> dict[str, str]:
        """Build mapping from model label IDs to standard NLI labels.

        Parameters
        ----------
        id2label
            Mapping from label IDs to label strings from model config.

        Returns
        -------
        dict[str, str]
            Mapping from label IDs (as strings) to standard labels.
        """
        mapping: dict[str, str] = {}
        for idx, label in id2label.items():
            # normalize label to lowercase
            normalized = label.lower()
            # map to standard labels
            if "entail" in normalized:
                mapping[str(idx)] = "entailment"
            elif "neutral" in normalized:
                mapping[str(idx)] = "neutral"
            elif "contradict" in normalized:
                mapping[str(idx)] = "contradiction"
            else:
                # keep original if we can't map it
                mapping[str(idx)] = normalized
        return mapping

    @property
    def model(self) -> PreTrainedModel:
        """Get the model, loading if necessary."""
        self._load_model()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer, loading if necessary."""
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text.

        Not supported for NLI models.

        Raises
        ------
        NotImplementedError
            Always raised, as NLI models don't provide log probabilities.
        """
        raise NotImplementedError(
            f"Log probability is not supported for NLI model {self.model_name}. "
            "Use HuggingFaceLanguageModel or HuggingFaceMaskedLanguageModel instead."
        )

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Not supported for NLI models.

        Raises
        ------
        NotImplementedError
            Always raised, as NLI models don't provide perplexity.
        """
        raise NotImplementedError(
            f"Perplexity is not supported for NLI model {self.model_name}. "
            "Use HuggingFaceLanguageModel or HuggingFaceMaskedLanguageModel instead."
        )

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Uses the model's encoder to get embeddings. Note that NLI models
        are typically fine-tuned for classification, so embeddings may not
        be optimal for general similarity tasks.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.
        """
        # check cache
        cached = self.cache.get(self.model_name, "embedding", text=text)
        if cached is not None:
            return cached

        # tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # get hidden states (using base model if available)
        with torch.no_grad():
            # try to access base model for embeddings
            if hasattr(self.model, "roberta"):
                base_model = self.model.roberta
            elif hasattr(self.model, "deberta"):
                base_model = self.model.deberta
            elif hasattr(self.model, "bert"):
                base_model = self.model.bert
            else:
                # fallback: use full model with output_hidden_states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states[0, 0].cpu().numpy()
                self.cache.set(
                    self.model_name,
                    "embedding",
                    embedding,
                    model_version=self.model_version,
                    text=text,
                )
                return embedding

            # use base model
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            # use [CLS] token
            embedding = outputs.last_hidden_state[0, 0].cpu().numpy()

        # cache result
        self.cache.set(
            self.model_name,
            "embedding",
            embedding,
            model_version=self.model_version,
            text=text,
        )

        return embedding

    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores.

        Parameters
        ----------
        premise : str
            Premise text.
        hypothesis : str
            Hypothesis text.

        Returns
        -------
        dict[str, float]
            Dictionary with keys "entailment", "neutral", "contradiction"
            mapping to probability scores that sum to ~1.0.
        """
        # check cache
        cached = self.cache.get(
            self.model_name, "nli", premise=premise, hypothesis=hypothesis
        )
        if cached is not None:
            return cached

        # tokenize premise-hypothesis pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # get logits
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]

        # convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        # map to standard labels
        scores: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            label = self._label_mapping.get(str(idx), str(idx))
            scores[label] = float(prob)

        # ensure we have all three standard labels
        for label in ["entailment", "neutral", "contradiction"]:
            if label not in scores:
                scores[label] = 0.0

        # cache result
        self.cache.set(
            self.model_name,
            "nli",
            scores,
            model_version=self.model_version,
            premise=premise,
            hypothesis=hypothesis,
        )

        return scores
