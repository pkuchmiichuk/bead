r"""Interpolation grammar and evaluator.

bead's config interpolation grammar matches OmegaConf's:

- ``${section.field}`` — absolute dotted-path reference.
- ``${.field}`` / ``${..field}`` — relative reference; each leading
  dot walks one level up from the current node's parent.
- ``${a.b[0]}`` / ``${a.b.0}`` — list indexing (bracketed or
  dotted-integer; both supported).
- ``${a.${b}}`` — nested interpolations; the inner is resolved first.
- ``"prefix_${a.b}_suffix"`` — string concatenation. A standalone
  ``${a.b}`` (whole-value, no surrounding text) substitutes the
  typed value; a substring substitution coerces to ``str``.
- ``${name:arg1,arg2}`` — resolver call. Built-in resolvers
  (``oc.env``, ``oc.select``, ``oc.dict.keys``, ``oc.dict.values``,
  ``oc.decode``, ``oc.deprecated``, ``oc.create``) are registered by
  :mod:`bead.config.compose.resolvers`; user code adds more via
  :func:`register_resolver`.
- ``\\${literal}`` — escape; produces a literal ``${literal}``.

Cycle detection raises :class:`~bead.config.compose.errors.InterpolationError`
with the cycle path in the message.

The evaluator operates on plain Python dicts and lists (and the
``ComposeValue`` union from :mod:`bead.data.base`). It imports nothing
from ``didactic`` or the rest of bead, so it can be lifted into a
standalone distribution without changes.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Final

from bead.config.compose.errors import InterpolationError

type ComposeValue = (
    str | int | float | bool | None | list["ComposeValue"] | dict[str, "ComposeValue"]
)
"""The kind of value the compose engine operates on.

This is the list-based JSON shape produced by ``yaml.safe_load`` and
``tomllib.load``. didactic validation accepts lists for ``tuple[T, ...]``
fields, so this type matches both inputs and validated outputs.

This alias lives in the subpackage so the package can be extracted
without depending on bead's ``ComposeValue``.
"""


ResolverFn = Callable[..., ComposeValue]
"""Type of a resolver function. Takes positional string args, returns
any JSON-shaped value. Resolvers may *themselves* contain
interpolations that are resolved before the resolver is called.
"""


_RESOLVERS: dict[str, ResolverFn] = {}

_ACTIVE_ROOT: ContextVar[dict[str, ComposeValue] | None] = ContextVar(
    "bead_compose_active_root", default=None
)
"""The dict currently being interpolated.

Set by :func:`resolve` for the duration of evaluation, so root-aware
resolvers (registered by application code) can reach the in-flight
config. :func:`active_root` is the public accessor.
"""


def active_root() -> dict[str, ComposeValue] | None:
    """Return the dict currently being interpolated, or ``None``.

    Resolvers that need to consult other parts of the composed config
    call this to retrieve the in-flight root. Returns ``None`` when
    called outside an active :func:`resolve` invocation.
    """
    return _ACTIVE_ROOT.get()


@contextmanager
def _activate_root(
    root: dict[str, ComposeValue],
) -> Generator[None]:
    token = _ACTIVE_ROOT.set(root)
    try:
        yield
    finally:
        _ACTIVE_ROOT.reset(token)


def register_resolver(name: str, fn: ResolverFn, *, replace: bool = False) -> None:
    """Register a custom resolver under ``name``.

    Parameters
    ----------
    name : str
        Resolver name as it appears in ``${name:args}``.
    fn : ResolverFn
        Function called with the comma-separated args (each a string,
        pre-stripped). The return value is substituted in place.
    replace : bool, optional
        Whether re-registration is allowed for an existing name.
        Defaults to ``False`` so accidental shadowing is loud.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``replace`` is ``False``.
    """
    if name in _RESOLVERS and not replace:
        raise ValueError(
            f"Resolver {name!r} already registered. Pass replace=True to override."
        )
    _RESOLVERS[name] = fn


def unregister_resolver(name: str) -> None:
    """Remove a registered resolver. No-op if it does not exist."""
    _RESOLVERS.pop(name, None)


def list_resolvers() -> tuple[str, ...]:
    """Return the names of every registered resolver, sorted."""
    return tuple(sorted(_RESOLVERS))


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Literal:
    """A literal string segment with no interpolation."""

    text: str


@dataclass(frozen=True)
class _Reference:
    """A reference of the form ``${path}``.

    ``up`` counts the number of leading dots (``${.x}`` → ``up=1``,
    ``${..x}`` → ``up=2``, ``${x}`` → ``up=0``). For absolute
    references ``up=0`` and the path starts at the root; relative
    references walk up that many parents before descending.

    Each element of ``parts`` is either a string (a dict key or a
    bare path segment) or an integer (a list index). String parts
    may themselves contain ``_Node`` lists, since interpolations can
    be nested inside path segments (``${a.${b.c}.d}``).
    """

    up: int
    parts: tuple[_PathSegment, ...]


@dataclass(frozen=True)
class _ResolverCall:
    """A resolver call of the form ``${name:arg1,arg2}``.

    ``args`` is a tuple of node-lists; each node-list is the AST for
    one argument, since arguments may themselves contain
    interpolations.
    """

    name: str
    args: tuple[tuple[_Node, ...], ...]


type _PathSegment = str | int | tuple["_Node", ...]
"""One element of an interpolation path.

A plain string or integer is a literal segment; a tuple-of-nodes is
a nested expression that must be evaluated to a string before being
spliced into the path.
"""


type _Node = _Literal | _Reference | _ResolverCall
"""One element of a parsed expression."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass
class _Parser:
    """Recursive-descent parser for interpolation strings.

    Maintains a single ``pos`` cursor into ``text``. Methods that
    parse a construct return its AST and advance the cursor.
    """

    text: str
    pos: int = 0

    def parse(self) -> tuple[_Node, ...]:
        """Parse the whole text into a node tuple."""
        return self._parse_until(end_chars=())

    def _parse_until(self, *, end_chars: tuple[str, ...]) -> tuple[_Node, ...]:
        """Parse until one of ``end_chars`` is reached (or EOF).

        Used both for the top level (``end_chars=()``) and for
        nested expression bodies.
        """
        nodes: list[_Node] = []
        buf: list[str] = []

        def flush_literal() -> None:
            if buf:
                nodes.append(_Literal("".join(buf)))
                buf.clear()

        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in end_chars:
                break
            if ch == "\\" and self.pos + 1 < len(self.text):
                nxt = self.text[self.pos + 1]
                if nxt == "$":
                    buf.append("$")
                    self.pos += 2
                    continue
                if nxt == "\\":
                    buf.append("\\")
                    self.pos += 2
                    continue
                buf.append(ch)
                self.pos += 1
                continue
            if ch == "$" and self._peek(1) == "{":
                flush_literal()
                nodes.append(self._parse_interp())
                continue
            buf.append(ch)
            self.pos += 1

        flush_literal()
        return tuple(nodes)

    def _peek(self, offset: int) -> str:
        idx = self.pos + offset
        if 0 <= idx < len(self.text):
            return self.text[idx]
        return ""

    def _parse_interp(self) -> _Node:
        """Parse a ``${...}`` expression starting at ``self.pos``."""
        assert self.text[self.pos : self.pos + 2] == "${"
        self.pos += 2  # consume "${"

        up = 0
        while self.pos < len(self.text) and self.text[self.pos] == ".":
            up += 1
            self.pos += 1

        parts: list[_PathSegment] = []
        head, head_terminator = self._parse_path_head()
        if head != "" or head_terminator == ":":
            parts.append(head)
        while head_terminator == ".":
            seg, head_terminator = self._parse_path_head()
            if seg == "":
                raise InterpolationError(
                    f"Empty path segment in interpolation at pos {self.pos}"
                )
            parts.append(seg)

        if head_terminator == ":":
            if up != 0:
                raise InterpolationError("Resolver call cannot have leading dots")
            if not parts or not all(isinstance(p, str) and p != "" for p in parts):
                raise InterpolationError(
                    "Resolver name must be a static dotted identifier"
                )
            name = ".".join(p for p in parts if isinstance(p, str))
            args = self._parse_resolver_args()
            self._expect("}")
            return _ResolverCall(name=name, args=args)

        while head_terminator == "[":
            self.pos += 1  # consume "["
            idx_text, _ = self._read_until("]")
            self._expect("]")
            try:
                parts.append(int(idx_text))
            except ValueError as exc:
                raise InterpolationError(
                    f"Bracketed index must be an integer: [{idx_text!r}]"
                ) from exc
            if self.pos < len(self.text) and self.text[self.pos] == ".":
                self.pos += 1
                seg, head_terminator = self._parse_path_head()
                if seg == "":
                    raise InterpolationError("Empty path segment after ']'")
                parts.append(seg)
            elif self.pos < len(self.text) and self.text[self.pos] == "[":
                self.pos += 1
                head_terminator = "["
                continue
            else:
                head_terminator = "}"

        if head_terminator != "}":
            raise InterpolationError(
                f"Unexpected character {head_terminator!r} in interpolation"
            )
        self.pos += 1  # consume "}"
        return _Reference(up=up, parts=tuple(parts))

    def _parse_path_head(self) -> tuple[_PathSegment, str]:
        """Read a single path segment, returning ``(segment, terminator)``.

        The terminator is one of ``"."`` (more dotted path follows),
        ``"["`` (bracketed index follows), ``":"`` (resolver call body
        follows), ``"}"`` (end of expression), or ``""`` (EOF).

        Segments may contain nested ``${...}`` interpolations, in
        which case the segment is returned as a node tuple instead
        of a string.
        """
        buf: list[str] = []
        nested: list[_Node] | None = None
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in (".", "[", ":", "}"):
                terminator = ch
                if ch == ".":
                    self.pos += 1
                break
            if ch == "$" and self._peek(1) == "{":
                if nested is None:
                    nested = []
                    if buf:
                        nested.append(_Literal("".join(buf)))
                        buf.clear()
                nested.append(self._parse_interp())
                continue
            buf.append(ch)
            self.pos += 1
        else:
            terminator = ""

        if nested is not None:
            if buf:
                nested.append(_Literal("".join(buf)))
            seg: _PathSegment = tuple(nested)
        else:
            seg = "".join(buf)
            if seg.lstrip("-").isdigit():
                seg = int(seg)
        return seg, terminator

    def _parse_resolver_args(self) -> tuple[tuple[_Node, ...], ...]:
        """Parse the body of a resolver call after the ':'."""
        assert self.text[self.pos] == ":"
        self.pos += 1
        args: list[tuple[_Node, ...]] = []
        current: list[_Node] = []
        buf: list[str] = []

        def flush_literal() -> None:
            if buf:
                current.append(_Literal("".join(buf)))
                buf.clear()

        depth = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "}" and depth == 0:
                flush_literal()
                args.append(tuple(current))
                return tuple(args)
            if ch == "," and depth == 0:
                flush_literal()
                args.append(tuple(current))
                current = []
                self.pos += 1
                continue
            if ch == "\\" and self.pos + 1 < len(self.text):
                buf.append(self.text[self.pos + 1])
                self.pos += 2
                continue
            if ch == "$" and self._peek(1) == "{":
                flush_literal()
                current.append(self._parse_interp())
                depth_at_call_start = depth
                _ = depth_at_call_start  # unused; documents intent
                continue
            buf.append(ch)
            self.pos += 1
        raise InterpolationError("Unterminated resolver call (missing '}')")

    def _read_until(self, terminator: str) -> tuple[str, str]:
        """Read literal text until ``terminator`` is the next char.

        Returns ``(text, terminator)``. Used for bracketed indices,
        which do not support nested interpolation.
        """
        start = self.pos
        while self.pos < len(self.text):
            if self.text[self.pos] == terminator:
                return self.text[start : self.pos], terminator
            self.pos += 1
        raise InterpolationError(f"Unterminated bracket: expected {terminator!r}")

    def _expect(self, ch: str) -> None:
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            raise InterpolationError(
                f"Expected {ch!r} at pos {self.pos}; "
                f"got {self.text[self.pos : self.pos + 1]!r}"
            )
        self.pos += 1


def _parse(text: str) -> tuple[_Node, ...]:
    """Public-internal parse entry point."""
    return _Parser(text).parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


_MAX_DEPTH: Final = 64


@dataclass
class _EvalState:
    """Mutable evaluator state threaded through recursive calls.

    Tracks the in-progress reference stack for cycle detection.
    """

    root: dict[str, ComposeValue]
    seen: set[tuple[int | str, ...]] = field(default_factory=set)


def resolve(
    node: ComposeValue,
    *,
    root: dict[str, ComposeValue],
    here: tuple[str | int, ...] = (),
) -> ComposeValue:
    """Resolve every interpolation in ``node``, returning a new value.

    Walks ``node`` recursively. Strings are parsed and evaluated;
    dicts and lists are descended into with ``here`` extended by the
    current key or index. Whole-value substitutions preserve type
    (the result of resolving ``"${a.b}"`` where ``a.b`` is an int is
    an int, not a stringified int); substring substitutions coerce
    to ``str``.

    Parameters
    ----------
    node : ComposeValue
        Value to resolve in place. Strings, dicts, lists, scalars.
    root : dict[str, ComposeValue]
        Top-level config; interpolations resolve against this.
    here : tuple[str | int, ...], optional
        Dotted path of ``node`` relative to ``root``. Used to resolve
        relative references (``${.x}``). Defaults to the empty path
        (top level).

    Returns
    -------
    ComposeValue
        Fully-resolved value, the same kind (str / int / dict / …)
        as ``node`` except where an interpolation changed the type.
    """
    state = _EvalState(root=root)
    with _activate_root(root):
        return _resolve_value(node, here, state)


def _resolve_value(
    node: ComposeValue,
    here: tuple[str | int, ...],
    state: _EvalState,
) -> ComposeValue:
    if isinstance(node, str):
        return _resolve_string(node, here, state)
    if isinstance(node, list):
        return [_resolve_value(item, (*here, i), state) for i, item in enumerate(node)]
    if isinstance(node, dict):
        return {
            key: _resolve_value(value, (*here, key), state)
            for key, value in node.items()
        }
    return node


def _resolve_string(
    text: str, here: tuple[str | int, ...], state: _EvalState
) -> ComposeValue:
    nodes = _parse(text)
    if len(nodes) == 1 and not isinstance(nodes[0], _Literal):
        value = _eval_node(nodes[0], here, state, depth=0)
        return value
    parts: list[str] = []
    for node in nodes:
        value = _eval_node(node, here, state, depth=0)
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(_to_str(value))
    return "".join(parts)


def _eval_node(
    node: _Node, here: tuple[str | int, ...], state: _EvalState, *, depth: int
) -> ComposeValue:
    if depth > _MAX_DEPTH:
        raise InterpolationError(f"Interpolation nesting exceeded {_MAX_DEPTH} levels")
    if isinstance(node, _Literal):
        return node.text
    if isinstance(node, _Reference):
        return _eval_reference(node, here, state, depth=depth)
    return _eval_resolver_call(node, here, state, depth=depth)


def _eval_reference(
    ref: _Reference,
    here: tuple[str | int, ...],
    state: _EvalState,
    *,
    depth: int,
) -> ComposeValue:
    if ref.up > len(here):
        raise InterpolationError(
            f"Relative reference {'.' * ref.up}... walks above the "
            f"root (current path: {_format_path(here)})"
        )
    base_path = here[: len(here) - ref.up] if ref.up > 0 else ()
    resolved_parts: list[str | int] = list(base_path)
    for part in ref.parts:
        resolved_parts.append(_resolve_path_segment(part, here, state, depth=depth + 1))

    cycle_key: tuple[int | str, ...] = (id(state.root),) + _hashable_path(
        resolved_parts
    )
    if cycle_key in state.seen:
        raise InterpolationError(
            f"Interpolation cycle detected at {_format_path(resolved_parts)}"
        )
    state.seen.add(cycle_key)
    try:
        value = _walk(state.root, resolved_parts)
        if isinstance(value, str | dict | list):
            return _resolve_value(value, tuple(resolved_parts), state)
        return value
    finally:
        state.seen.discard(cycle_key)


def _eval_resolver_call(
    call: _ResolverCall,
    here: tuple[str | int, ...],
    state: _EvalState,
    *,
    depth: int,
) -> ComposeValue:
    if call.name not in _RESOLVERS:
        raise InterpolationError(
            f"Unknown resolver {call.name!r}. Registered: {list_resolvers()}"
        )
    resolved_args: list[str] = []
    for arg_nodes in call.args:
        if not arg_nodes:
            resolved_args.append("")
            continue
        if len(arg_nodes) == 1 and isinstance(arg_nodes[0], _Literal):
            resolved_args.append(arg_nodes[0].text)
            continue
        rendered: list[str] = []
        for sub in arg_nodes:
            value = _eval_node(sub, here, state, depth=depth + 1)
            rendered.append(value if isinstance(value, str) else _to_str(value))
        resolved_args.append("".join(rendered))

    try:
        return _RESOLVERS[call.name](*resolved_args)
    except InterpolationError:
        raise
    except Exception as exc:
        raise InterpolationError(
            f"Resolver {call.name!r} raised {type(exc).__name__}: {exc}"
        ) from exc


def _resolve_path_segment(
    seg: _PathSegment,
    here: tuple[str | int, ...],
    state: _EvalState,
    *,
    depth: int,
) -> str | int:
    if isinstance(seg, str | int):
        return seg
    rendered: list[str] = []
    for sub in seg:
        value = _eval_node(sub, here, state, depth=depth)
        rendered.append(value if isinstance(value, str) else _to_str(value))
    joined = "".join(rendered)
    if joined.lstrip("-").isdigit():
        return int(joined)
    return joined


def _walk(
    root: ComposeValue, path: list[str | int] | tuple[str | int, ...]
) -> ComposeValue:
    cur: ComposeValue = root
    for i, part in enumerate(path):
        if isinstance(cur, dict):
            if not isinstance(part, str):
                raise InterpolationError(
                    f"Cannot index dict at {_format_path(path[: i + 1])} with integer"
                )
            if part not in cur:
                raise InterpolationError(
                    f"Reference {_format_path(path[: i + 1])} unresolved"
                )
            cur = cur[part]
        elif isinstance(cur, list):
            if not isinstance(part, int):
                raise InterpolationError(
                    f"Cannot index list at "
                    f"{_format_path(path[: i + 1])} with non-integer"
                )
            if part < 0 or part >= len(cur):
                raise InterpolationError(
                    f"List index out of range at "
                    f"{_format_path(path[: i + 1])} (len={len(cur)})"
                )
            cur = cur[part]
        else:
            raise InterpolationError(
                f"Cannot descend into scalar at {_format_path(path[:i])}"
            )
    return cur


def _hashable_path(path: list[str | int]) -> tuple[str | int, ...]:
    return tuple(path)


def _format_path(path: list[str | int] | tuple[str | int, ...]) -> str:
    if not path:
        return "<root>"
    parts: list[str] = []
    for p in path:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            parts.append(f".{p}" if parts else p)
    return "".join(parts)


def _to_str(value: ComposeValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
