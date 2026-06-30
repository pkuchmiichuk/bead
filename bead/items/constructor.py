"""Item constructor for building experimental items from templates.

This module provides the ItemConstructor class which transforms filled templates
into experimental items by applying model-based constraints and collecting
model outputs for analysis.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID

import numpy as np

from bead.dsl.ast import (
    ASTNode,
    AttributeAccess,
    BinaryOp,
    FunctionCall,
    Literal,
    UnaryOp,
    Variable,
)
from bead.dsl.context import EvaluationContext
from bead.dsl.evaluator import Evaluator
from bead.dsl.parser import parse
from bead.dsl.stdlib import register_stdlib
from bead.items.adapters.registry import ModelAdapterRegistry
from bead.items.cache import ModelOutputCache
from bead.items.item import Item, MetadataValue, ModelOutput
from bead.items.item_template import ItemTemplate
from bead.resources.constraints import Constraint
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver


class ItemConstructor:
    """Construct experimental items from filled templates.

    Transforms filled templates into items by:
    1. Resolving element references to text
    2. Computing required model outputs (from constraints)
    3. Evaluating constraints with model outputs
    4. Creating Item instances with metadata

    Parameters
    ----------
    model_registry : ModelAdapterRegistry
        Registry of model adapters for constraint evaluation.
    cache : ModelOutputCache
        Cache for model outputs to avoid redundant computation.
    constraint_resolver : ConstraintResolver | None, optional
        Resolver for evaluating non-model constraints. If None, only
        model-based constraints can be evaluated.

    Attributes
    ----------
    model_registry : ModelAdapterRegistry
        Registry of model adapters for constraint evaluation.
    cache : ModelOutputCache
        Cache for model outputs to avoid redundant computation.
    constraint_resolver : ConstraintResolver | None
        Resolver for evaluating constraints (not used for model constraints).

    Examples
    --------
    >>> from bead.items.adapters.registry import default_registry
    >>> from bead.items.cache import ModelOutputCache
    >>> cache = ModelOutputCache(backend="memory")
    >>> constructor = ItemConstructor(default_registry, cache)
    >>> constraints = {constraint_id: constraint_obj}
    >>> items = list(constructor.construct_items(
    ...     template, filled_templates, constraints
    ... ))
    """

    def __init__(
        self,
        model_registry: ModelAdapterRegistry,
        cache: ModelOutputCache,
        constraint_resolver: ConstraintResolver | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.cache = cache
        self.constraint_resolver = constraint_resolver
        self._dsl_evaluator = Evaluator(use_cache=True)

    def construct_items(
        self,
        item_template: ItemTemplate,
        filled_templates: dict[UUID, FilledTemplate],
        constraints: dict[UUID, Constraint],
    ) -> Iterator[Item]:
        """Construct items from template and filled templates.

        For each combination of filled templates:
        1. Render elements (resolve filled_template_ref → text)
        2. Compute required model outputs (from constraints)
        3. Check constraints using model outputs
        4. Yield item if all constraints satisfied

        Parameters
        ----------
        item_template : ItemTemplate
            Template defining item structure and constraints.
        filled_templates : dict[UUID, FilledTemplate]
            Map of filled template UUIDs to FilledTemplate instances.
        constraints : dict[UUID, Constraint]
            Map of constraint UUIDs to Constraint objects.

        Yields
        ------
        Item
            Constructed items that satisfy all constraints.

        Raises
        ------
        ValueError
            If template references missing filled templates or constraints.
        RuntimeError
            If constraint evaluation or model computation fails.

        Examples
        --------
        >>> template = ItemTemplate(...)
        >>> filled = {uuid1: filled1, uuid2: filled2}
        >>> constraints = {c_id: constraint_obj}
        >>> items = list(constructor.construct_items(
        ...     template, filled, constraints
        ... ))
        >>> len(items)
        2
        """
        # Render elements to text
        rendered_elements = self._render_elements(item_template, filled_templates)

        # Compute model outputs required by constraints
        model_outputs = self._compute_model_outputs(
            item_template, rendered_elements, constraints
        )

        # Check constraints
        constraint_satisfaction = self._check_constraints(
            item_template, rendered_elements, model_outputs, constraints
        )

        # Only yield item if all constraints satisfied
        if all(constraint_satisfaction.values()):
            from bead.items.item import ConstraintSatisfaction  # noqa: PLC0415

            item = Item(
                item_template_id=item_template.id,
                filled_template_refs=tuple(filled_templates.keys()),
                rendered_elements=rendered_elements,
                model_outputs=tuple(model_outputs),
                constraint_satisfaction=tuple(
                    ConstraintSatisfaction(constraint_id=cid, satisfied=ok)
                    for cid, ok in constraint_satisfaction.items()
                ),
            )
            yield item

    def _render_elements(
        self,
        item_template: ItemTemplate,
        filled_templates: dict[UUID, FilledTemplate],
    ) -> dict[str, str]:
        """Render ItemElements to text.

        Resolve element references: text elements use content directly,
        filled_template_ref elements use the rendered text from FilledTemplate.

        Parameters
        ----------
        item_template : ItemTemplate
            Template with elements to render.
        filled_templates : dict[UUID, FilledTemplate]
            Map of filled template UUIDs to instances.

        Returns
        -------
        dict[str, str]
            Map of element names to rendered text.

        Raises
        ------
        ValueError
            If element references missing filled template.
        """
        rendered: dict[str, str] = {}

        for element in item_template.elements:
            if element.is_text:
                # Static text element
                rendered[element.element_name] = element.content or ""
            elif element.is_template_ref:
                # Reference to filled template
                ref_id = element.filled_template_ref_id
                if ref_id is None:
                    raise ValueError(
                        f"Element {element.element_name} has no filled_template_ref_id"
                    )
                if ref_id not in filled_templates:
                    raise ValueError(
                        f"Element {element.element_name} references missing "
                        f"filled template {ref_id}"
                    )
                filled_template = filled_templates[ref_id]
                rendered[element.element_name] = filled_template.rendered_text

        return rendered

    def _compute_model_outputs(
        self,
        item_template: ItemTemplate,
        rendered_elements: dict[str, str],
        constraints: dict[UUID, Constraint],
    ) -> list[ModelOutput]:
        """Execute model calls required by constraints.

        Parse DSL constraints to find model function calls, then execute
        them via adapters with caching.

        Parameters
        ----------
        item_template : ItemTemplate
            Template with constraints.
        rendered_elements : dict[str, str]
            Rendered element text.
        constraints : dict[UUID, Constraint]
            Map of constraint UUIDs to Constraint objects.

        Returns
        -------
        list[ModelOutput]
            All model outputs computed for this item.

        Raises
        ------
        RuntimeError
            If model computation fails.
        ValueError
            If constraint UUID not found in constraints dict.
        """
        model_outputs: list[ModelOutput] = []

        # Extract model calls from all DSL constraints
        for constraint_id in item_template.constraints:
            if constraint_id not in constraints:
                raise ValueError(f"Constraint {constraint_id} not found")

            constraint = constraints[constraint_id]

            # Parse constraint expression to AST
            try:
                ast_node = parse(constraint.expression)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse constraint '{constraint.expression}': {e}"
                ) from e

            # Extract all model function calls from AST
            model_calls = self._extract_model_calls(ast_node, rendered_elements)

            # Execute each model call
            for call in model_calls:
                try:
                    output = self._execute_model_call(call)
                    if output:
                        model_outputs.append(output)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to execute model call {call}: {e}"
                    ) from e

        return model_outputs

    def _extract_model_calls(
        self, ast_node: ASTNode, rendered_elements: dict[str, str]
    ) -> list[dict[str, str | int | float | bool | None]]:
        """Extract model function calls from AST.

        Recursively traverse AST to find calls to model functions
        (lm_prob, nli, similarity, etc.) and extract their arguments.

        Parameters
        ----------
        ast_node : ASTNode
            AST node to traverse.
        rendered_elements : dict[str, str]
            Rendered elements for variable resolution.

        Returns
        -------
        list[dict[str, str | int | float | bool | None]]
            List of model call specifications with function name and arguments.
        """
        calls: list[dict[str, str | int | float | bool | None]] = []

        if isinstance(ast_node, FunctionCall):
            # Check if this is a model function call
            # Function can be Variable (for functions) or AttributeAccess (for methods)
            if isinstance(ast_node.function, Variable):
                func_name: str = ast_node.function.name
            elif isinstance(ast_node.function, AttributeAccess):
                func_name = ast_node.function.attribute
            else:
                # Skip other function call types
                return calls

            model_functions = {
                "lm_prob",
                "lm_perplexity",
                "nli",
                "similarity",
                "embedding",
            }
            if func_name in model_functions:
                # Extract arguments
                call_spec = self._extract_call_args(
                    func_name, ast_node.arguments, rendered_elements
                )
                if call_spec:
                    calls.append(call_spec)

            # Also check arguments for nested calls
            for arg in ast_node.arguments:
                calls.extend(self._extract_model_calls(arg, rendered_elements))

        # Recursively check other node types
        elif isinstance(ast_node, BinaryOp):
            calls.extend(self._extract_model_calls(ast_node.left, rendered_elements))
            calls.extend(self._extract_model_calls(ast_node.right, rendered_elements))
        elif isinstance(ast_node, UnaryOp):
            calls.extend(self._extract_model_calls(ast_node.operand, rendered_elements))
        elif isinstance(ast_node, AttributeAccess):
            calls.extend(self._extract_model_calls(ast_node.object, rendered_elements))

        return calls

    def _extract_call_args(
        self,
        func_name: str,
        args: list[ASTNode] | tuple[ASTNode, ...],
        rendered_elements: dict[str, str],
    ) -> dict[str, str | int | float | bool | None] | None:
        """Extract arguments from a model function call.

        Parameters
        ----------
        func_name : str
            Name of the function.
        args : list[ASTNode]
            AST nodes representing function arguments.
        rendered_elements : dict[str, str]
            Rendered elements for variable resolution.

        Returns
        -------
        dict[str, Any] | None
            Call specification with function, args, and model name.
        """
        # Resolve literal values and variables
        resolved_args: list[str | int | float | bool | None] = []
        for arg in args:
            if isinstance(arg, Literal):
                resolved_args.append(arg.value)
            elif isinstance(arg, Variable):
                # Try to resolve from rendered elements
                if arg.name in rendered_elements:
                    resolved_args.append(rendered_elements[arg.name])
                else:
                    # Can't resolve, skip this call
                    return None
            else:
                # Complex expression, can't extract statically
                return None

        # Build call specification based on function type
        if func_name in {"lm_prob", "lm_perplexity"}:
            # lm_prob(text, model='gpt2')
            if len(resolved_args) == 0:
                return None
            text = str(resolved_args[0])
            model = str(resolved_args[1]) if len(resolved_args) > 1 else "gpt2"
            operation = "log_probability" if func_name == "lm_prob" else "perplexity"
            return {
                "function": func_name,
                "text": text,
                "model": model,
                "operation": operation,
            }

        elif func_name == "nli":
            # nli(premise, hypothesis, model='roberta-large-mnli')
            if len(resolved_args) < 2:
                return None
            premise = str(resolved_args[0])
            hypothesis = str(resolved_args[1])
            default_nli_model = "roberta-large-mnli"
            model = (
                str(resolved_args[2]) if len(resolved_args) > 2 else default_nli_model
            )
            return {
                "function": func_name,
                "premise": premise,
                "hypothesis": hypothesis,
                "model": model,
                "operation": "nli",
            }

        elif func_name == "similarity":
            # similarity(text1, text2, model='all-MiniLM-L6-v2')
            if len(resolved_args) < 2:
                return None
            text1 = str(resolved_args[0])
            text2 = str(resolved_args[1])
            model = (
                str(resolved_args[2]) if len(resolved_args) > 2 else "all-MiniLM-L6-v2"
            )
            return {
                "function": func_name,
                "text1": text1,
                "text2": text2,
                "model": model,
                "operation": "similarity",
            }

        elif func_name == "embedding":
            # embedding(text, model='all-MiniLM-L6-v2')
            if len(resolved_args) == 0:
                return None
            text = str(resolved_args[0])
            model = (
                str(resolved_args[1]) if len(resolved_args) > 1 else "all-MiniLM-L6-v2"
            )
            return {
                "function": func_name,
                "text": text,
                "model": model,
                "operation": "embedding",
            }

        return None

    def _execute_model_call(
        self, call_spec: dict[str, str | int | float | bool | None]
    ) -> ModelOutput | None:
        """Execute a single model call and return ModelOutput.

        Parameters
        ----------
        call_spec : dict[str, str | int | float | bool | None]
            Call specification with function, args, and model.

        Returns
        -------
        ModelOutput | None
            Model output if successful, None if already cached or failed.

        Raises
        ------
        RuntimeError
            If model execution fails.
        """
        operation = str(call_spec["operation"])
        model_name = str(call_spec["model"])

        # Determine adapter type based on operation
        if operation in {"log_probability", "perplexity"}:
            adapter_type = "huggingface_lm"
        elif operation == "nli":
            adapter_type = "huggingface_nli"
        elif operation in {"similarity", "embedding"}:
            adapter_type = "sentence_transformer"
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Check cache first
        cache_key_args: dict[str, str | int | float | bool | None] = {}
        if operation in {"log_probability", "perplexity"}:
            cache_key_args = {"text": call_spec["text"]}
        elif operation == "nli":
            cache_key_args = {
                "premise": call_spec["premise"],
                "hypothesis": call_spec["hypothesis"],
            }
        elif operation == "similarity":
            cache_key_args = {
                "text1": call_spec["text1"],
                "text2": call_spec["text2"],
            }
        elif operation == "embedding":
            cache_key_args = {"text": call_spec["text"]}

        cached_result = self.cache.get(model_name, operation, **cache_key_args)
        if cached_result is not None:
            # Already cached, create ModelOutput from cache
            cache_key = self.cache.generate_cache_key(
                model_name, operation, **cache_key_args
            )
            # Convert inputs to MetadataValue compatible dict
            metadata_inputs: dict[str, MetadataValue] = {
                k: str(v) for k, v in cache_key_args.items()
            }
            return ModelOutput(
                model_name=model_name,
                model_version="unknown",  # Could fetch from cache
                operation=operation,
                inputs=metadata_inputs,
                output=cached_result,
                cache_key=cache_key,
                computation_metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "from_cache": True,
                },
            )

        # Get adapter and execute
        adapter = self.model_registry.get_adapter(
            adapter_type=adapter_type,
            model_name=model_name,
            cache=self.cache,
        )

        # Execute the operation
        if operation == "log_probability":
            result = adapter.compute_log_probability(str(call_spec["text"]))
        elif operation == "perplexity":
            result = adapter.compute_perplexity(str(call_spec["text"]))
        elif operation == "nli":
            result = adapter.compute_nli(
                str(call_spec["premise"]), str(call_spec["hypothesis"])
            )
        elif operation == "similarity":
            result = adapter.compute_similarity(
                str(call_spec["text1"]), str(call_spec["text2"])
            )
        elif operation == "embedding":
            result = adapter.get_embedding(str(call_spec["text"]))
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Generate cache key
        cache_key = self.cache.generate_cache_key(
            model_name, operation, **cache_key_args
        )

        # Convert inputs to MetadataValue compatible dict
        metadata_inputs: dict[str, MetadataValue] = {
            k: str(v) for k, v in cache_key_args.items()
        }

        # Create ModelOutput
        model_version = (
            adapter.model_version if hasattr(adapter, "model_version") else "unknown"
        )
        return ModelOutput(
            model_name=model_name,
            model_version=model_version,
            operation=operation,
            inputs=metadata_inputs,
            output=result,  # type: ignore[arg-type]  # Output can be various types
            cache_key=cache_key,
            computation_metadata={
                "timestamp": datetime.now(UTC).isoformat(),
                "from_cache": False,
            },
        )

    def _check_constraints(
        self,
        item_template: ItemTemplate,
        rendered_elements: dict[str, str],
        model_outputs: list[ModelOutput],
        constraints: dict[UUID, Constraint],
    ) -> dict[UUID, bool]:
        """Evaluate constraints using model outputs.

        Check each constraint against rendered elements and model outputs.

        Parameters
        ----------
        item_template : ItemTemplate
            Template with constraints.
        rendered_elements : dict[str, str]
            Rendered element text.
        model_outputs : list[ModelOutput]
            Model outputs to use in constraint evaluation.
        constraints : dict[UUID, Constraint]
            Map of constraint UUIDs to Constraint objects.

        Returns
        -------
        dict[UUID, bool]
            Map of constraint UUIDs to satisfaction status.

        Raises
        ------
        RuntimeError
            If constraint evaluation fails.
        ValueError
            If constraint UUID not found.
        """
        constraint_satisfaction: dict[UUID, bool] = {}

        # Evaluate each constraint
        for constraint_id in item_template.constraints:
            if constraint_id not in constraints:
                raise ValueError(f"Constraint {constraint_id} not found")

            constraint = constraints[constraint_id]

            # Evaluate constraint
            satisfied = self._evaluate_dsl_constraint(
                constraint, rendered_elements, model_outputs
            )
            constraint_satisfaction[constraint_id] = satisfied

        return constraint_satisfaction

    def _evaluate_dsl_constraint(
        self,
        constraint: Constraint,
        rendered_elements: dict[str, str],
        model_outputs: list[ModelOutput],
    ) -> bool:
        """Evaluate a DSL constraint with model outputs.

        Parse and evaluate DSL expression with element variables and
        model output values in context.

        Parameters
        ----------
        constraint : Constraint
            Constraint to evaluate.
        rendered_elements : dict[str, str]
            Rendered element text for variable substitution.
        model_outputs : list[ModelOutput]
            Model outputs to include in context.

        Returns
        -------
        bool
            True if constraint is satisfied.

        Raises
        ------
        RuntimeError
            If DSL evaluation fails.
        """
        # Create evaluation context
        context = EvaluationContext()

        # Register standard library
        register_stdlib(context)

        # Register model functions that will use cached outputs
        self._register_model_functions(context, model_outputs)

        # Set element variables
        for name, text in rendered_elements.items():
            context.set_variable(name, text)

        # Parse and evaluate
        try:
            ast_node = parse(constraint.expression)
            result = self._dsl_evaluator.evaluate(ast_node, context)
            return bool(result)
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate constraint '{constraint.expression}': {e}"
            ) from e

    def _register_model_functions(
        self,
        context: EvaluationContext,
        model_outputs: list[ModelOutput],
    ) -> None:
        """Register model functions in DSL context.

        Add functions like lm_prob(), nli(), similarity() that can access
        precomputed model outputs from cache.

        Parameters
        ----------
        context : EvaluationContext
            DSL evaluation context.
        model_outputs : list[ModelOutput]
            Precomputed model outputs.
        """
        # Create lookup for model outputs
        output_map: dict[tuple[str, str, str], ModelOutput] = {}
        for output in model_outputs:
            # Key includes model, operation, and stringified inputs
            inputs_str = str(sorted(output.inputs.items()))
            key = (output.model_name, output.operation, inputs_str)
            output_map[key] = output

        # Define model functions that use cached outputs
        def lm_prob(text: str, model: str = "gpt2") -> float:
            """Get log probability from cache or compute."""
            # Check cache first
            cached = self.cache.get(model, "log_probability", text=text)
            if cached is not None:
                return float(cached)

            # Compute if not cached
            adapter = self.model_registry.get_adapter(
                adapter_type="huggingface_lm",
                model_name=model,
                cache=self.cache,
            )
            result = adapter.compute_log_probability(text)
            return result

        def lm_perplexity(text: str, model: str = "gpt2") -> float:
            """Get perplexity from cache or compute."""
            cached = self.cache.get(model, "perplexity", text=text)
            if cached is not None:
                return float(cached)

            adapter = self.model_registry.get_adapter(
                adapter_type="huggingface_lm",
                model_name=model,
                cache=self.cache,
            )
            result = adapter.compute_perplexity(text)
            return result

        def nli(
            premise: str, hypothesis: str, model: str = "roberta-large-mnli"
        ) -> dict[str, float]:
            """Get NLI scores from cache or compute."""
            cached = self.cache.get(
                model, "nli", premise=premise, hypothesis=hypothesis
            )
            if cached is not None:
                return dict(cached)  # type: ignore[arg-type]

            adapter = self.model_registry.get_adapter(
                adapter_type="huggingface_nli",
                model_name=model,
                cache=self.cache,
            )
            result = adapter.compute_nli(premise, hypothesis)
            return result

        def similarity(
            text1: str, text2: str, model: str = "all-MiniLM-L6-v2"
        ) -> float:
            """Get similarity from cache or compute."""
            cached = self.cache.get(model, "similarity", text1=text1, text2=text2)
            if cached is not None:
                return float(cached)

            adapter = self.model_registry.get_adapter(
                adapter_type="sentence_transformer",
                model_name=model,
                cache=self.cache,
            )
            result = adapter.compute_similarity(text1, text2)
            return result

        def embedding(text: str, model: str = "all-MiniLM-L6-v2") -> list[float]:
            """Get embedding from cache or compute."""
            cached = self.cache.get(model, "embedding", text=text)
            if cached is not None:
                # Convert numpy array back to list
                if isinstance(cached, np.ndarray):
                    return cached.tolist()  # type: ignore[return-value]
                return list(cached)  # type: ignore[arg-type]

            adapter = self.model_registry.get_adapter(
                adapter_type="sentence_transformer",
                model_name=model,
                cache=self.cache,
            )
            result = adapter.get_embedding(text)
            return result.tolist()  # type: ignore[return-value]

        # Register functions in context
        context.set_function("lm_prob", lm_prob)
        context.set_function("lm_perplexity", lm_perplexity)
        context.set_function("nli", nli)
        context.set_function("similarity", similarity)
        context.set_function("embedding", embedding)
