"""ADR-0276: every provider call outside ``zakcode.providers`` is one the live status file sees.

A side call is marked by hand where it is made (``_status_writer().side_call(...)``). A call
added later without a mark reads in the status file as whatever state came before it, which is
the 11-minute "working" misread ADR-0276 was written for, and no other test fails. This walks
``src/`` with :mod:`ast` and requires every use of a provider completion method to be one of:

* a conversation call, pinned by name in :data:`CONVERSATION`, which its function marks with
  ``set_model_call_start`` before making it;
* inside a ``side_call`` block, at the call itself or at every place the function making it is
  used, followed up through the callers (a nested function within the function around it, a
  method by attribute, a module function by name or attribute anywhere in ``src/``);
* reached only from an entry point in :data:`LIBRARY_ONLY`, each listed with its reason, and
  each reason checked so an entry cannot outlive it.

A function counts as used wherever its name is read, called or not, so a helper handed over as
a callback (the compaction's retried request) is followed to where it is handed over. Names are
resolved lexically, which errs strict: a same-named function elsewhere adds uses that must be
marked too. A call reached through ``getattr`` or a table of callables built at run time is
beyond what this can see.
"""

from __future__ import annotations

import ast
import functools
import inspect
from dataclasses import dataclass
from pathlib import Path

from zakcode.providers.base import Provider

SRC = Path(__file__).resolve().parents[1] / "src"
#: The providers themselves make the calls this file is about; their own calls are not uses.
PROVIDERS = "zakcode/providers/"

#: Provider methods that run a completion: its async methods (``acomplete``, ``astream``).
COMPLETION_METHODS: frozenset[str] = frozenset(
    name
    for name, member in vars(Provider).items()
    if inspect.iscoroutinefunction(member) or inspect.isasyncgenfunction(member)
)

#: The conversation's own calls (ADR-0266), by function. Each holds exactly one provider call and
#: marks it with ``set_model_call_start`` first.
CONVERSATION: dict[str, str] = {
    "zakcode/agent/loop.py::AgentLoop._call_provider.complete": "a buffered turn's model call",
    "zakcode/agent/loop.py::AgentLoop.astream_turn": "a streamed turn's model call",
}

#: Entry points only the library API reaches, so no status file can be written while they run:
#: ``zakcode cli`` is the only process that writes one. Each is ``(reason, option)``. With no
#: option, nothing in ``src/`` may use the entry point at all. With one, the entry point is built
#: only when ``Agent`` is given that option, and nothing outside ``zakcode/__init__.py`` may name
#: it, so the CLI cannot be passing it.
LIBRARY_ONLY: dict[str, tuple[str, str | None]] = {
    "zakcode/context/classify.py::SmallModelClassifier.__call__": (
        "the context relevance classifier, built only by Agent(context_classifier='model')",
        "context_classifier",
    ),
    "zakcode/context/used.py::ModelUsedDetector.__call__": (
        "the context-use judge, built only by Agent(context_signal_judge=True)",
        "context_signal_judge",
    ),
    "zakcode/quality/judge.py::vote_binary": ("a judge panel exported for library callers", None),
    "zakcode/quality/bestof.py::best_of_n": (
        "best-of-n sampling exported for library callers",
        None,
    ),
    "zakcode/quality/plan.py::judge_plan": (
        "pairwise plan choice exported for library callers",
        None,
    ),
    "zakcode/quality/select.py::select_best": (
        "oracle-then-judge choice for library callers",
        None,
    ),
    "zakcode/quality/refine.py::refine": ("score-and-refine exported for library callers", None),
}
_OPTIONS = frozenset(option for _, option in LIBRARY_ONLY.values() if option)
#: The library facade, where ``Agent`` takes the options above.
_FACADE = "zakcode/__init__.py"

_HOW_TO_FIX = (
    "Mark each: wrap the call, or every use of the function making it, in "
    "`_status_writer().side_call(session_id, '<kind>')` (ADR-0276). If only the library API "
    "reaches it, add its entry point to LIBRARY_ONLY in tests/test_provider_calls_marked.py "
    "with the reason."
)


@dataclass(frozen=True)
class Use:
    """One place a name is read: its file and line, the innermost function around it (``None``
    at module or class level), and whether a ``side_call`` block in that function holds it."""

    rel: str
    line: int
    fn: str | None
    marked: bool


@dataclass(frozen=True)
class Fn:
    """A function, keyed ``"<path under src>::<qualname>"``. ``parent`` is the function it is
    nested in; ``method`` says it is defined directly in a class body."""

    key: str
    name: str
    parent: str | None
    method: bool


def _within(fn: str, outer: str) -> bool:
    """Whether function key ``fn`` is ``outer`` or nested anywhere inside it."""
    return fn == outer or fn.startswith(outer + ".")


def _called_name(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else None


class Scan:
    """What the source says, for the checks below. ``overrides`` replaces a file's text by its
    path under ``src/``, which is how the positive control unmarks a call without touching disk."""

    def __init__(self, src: Path, overrides: dict[str, str] | None = None) -> None:
        self.fns: dict[str, Fn] = {}
        self.names: dict[str, list[Use]] = {}
        self.attrs: dict[str, list[Use]] = {}
        #: Each use of a completion method outside the providers, with the method's name.
        self.provider: list[tuple[Use, str]] = []
        #: Lines of ``set_model_call_start`` reads, by function.
        self.marks: dict[str, list[int]] = {}
        #: Files that name an option of :data:`LIBRARY_ONLY`, by option.
        self.option_files: dict[str, set[str]] = {}
        for path in sorted((src / "zakcode").rglob("*.py")):
            rel = path.relative_to(src).as_posix()
            text = (overrides or {}).get(rel)
            if text is None:
                text = path.read_text(encoding="utf-8")
            _Visitor(self, rel).visit(ast.parse(text, filename=rel))

    def uses_of(self, fn: Fn) -> list[Use]:
        """Where ``fn``'s name is read, leaving out its own body (recursion adds no way in)."""
        if fn.parent is not None:
            # A nested function's name exists only inside the function around it.
            found = [u for u in self.names.get(fn.name, []) if u.fn and _within(u.fn, fn.parent)]
        elif fn.method:
            found = list(self.attrs.get(fn.name, []))
        else:
            found = self.names.get(fn.name, []) + self.attrs.get(fn.name, [])
        return [u for u in found if not (u.fn and _within(u.fn, fn.key))]

    def unmarked(self, key: str, seen: frozenset[str] = frozenset()) -> list[str] | None:
        """``None`` when every way into function ``key`` is marked (or library-only). Else the
        unmarked uses from ``key`` up, each ``"<file:line> in <function>"``, and last why the
        trail ends: nothing uses the function, a use outside any function, or a cycle."""
        if key in LIBRARY_ONLY:
            return None
        uses = self.uses_of(self.fns[key])
        if not uses:
            return ["(used nowhere in src/)"]
        for use in uses:
            if use.marked:
                continue
            here = f"{use.rel}:{use.line}"
            if use.fn is None:
                return [f"{here} (outside any function)"]
            if use.fn in seen:
                return [f"{here} in {use.fn} (a cycle)"]
            chain = self.unmarked(use.fn, seen | {key})
            if chain is not None:
                return [f"{here} in {use.fn}", *chain]
        return None


class _Visitor(ast.NodeVisitor):
    """Records functions, name reads, completion-method reads and ``side_call`` blocks for one
    module. A ``side_call`` block counts only within the function it is written in."""

    def __init__(self, scan: Scan, rel: str) -> None:
        self.scan = scan
        self.rel = rel
        self.quals: list[str] = []
        self.fn: str | None = None
        self.in_class = False
        self.marked = 0

    def _use(self, line: int) -> Use:
        return Use(self.rel, line, self.fn, self.marked > 0)

    def _option(self, word: str | None) -> None:
        if word in _OPTIONS:
            self.scan.option_files.setdefault(word, set()).add(self.rel)

    def _def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Decorators, defaults and annotations run in the enclosing scope.
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self.quals.append(node.name)
        key = f"{self.rel}::{'.'.join(self.quals)}"
        self.scan.fns[key] = Fn(key, node.name, self.fn, self.in_class)
        saved = (self.fn, self.in_class, self.marked)
        self.fn, self.in_class, self.marked = key, False, 0
        for stmt in node.body:
            self.visit(stmt)
        self.fn, self.in_class, self.marked = saved
        self.quals.pop()

    visit_FunctionDef = _def
    visit_AsyncFunctionDef = _def

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expr in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expr)
        self.quals.append(node.name)
        saved = (self.fn, self.in_class, self.marked)
        self.fn, self.in_class, self.marked = None, True, 0
        for stmt in node.body:
            self.visit(stmt)
        self.fn, self.in_class, self.marked = saved
        self.quals.pop()

    def _with(self, node: ast.With | ast.AsyncWith) -> None:
        opens = any(_called_name(item.context_expr) == "side_call" for item in node.items)
        for item in node.items:
            self.visit(item)
        self.marked += opens
        for stmt in node.body:
            self.visit(stmt)
        self.marked -= opens

    visit_With = _with
    visit_AsyncWith = _with

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.scan.names.setdefault(node.id, []).append(self._use(node.lineno))
        self._option(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            use = self._use(node.lineno)
            self.scan.attrs.setdefault(node.attr, []).append(use)
            if node.attr in COMPLETION_METHODS and not self.rel.startswith(PROVIDERS):
                self.scan.provider.append((use, node.attr))
            if node.attr == "set_model_call_start" and self.fn is not None:
                self.scan.marks.setdefault(self.fn, []).append(node.lineno)
        self._option(node.attr)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._option(node.arg)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        self._option(node.arg)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._option(node.value)


@functools.cache
def _scan() -> Scan:
    return Scan(SRC)


def unmarked_calls(scan: Scan) -> list[str]:
    """One line per provider call the status file cannot see, with the chain that shows it."""
    problems = []
    for use, method in scan.provider:
        if use.marked or use.fn in CONVERSATION:
            continue
        where = f"{use.rel}:{use.line} .{method}()"
        if use.fn is None:
            problems.append(f"{where}, outside any function")
            continue
        chain = scan.unmarked(use.fn)
        if chain is not None:
            problems.append("\n      <- ".join([f"{where} in {use.fn}", *chain]))
    return problems


def _report(problems: list[str]) -> str:
    return (
        "provider calls the live status file cannot see:\n  "
        + "\n  ".join(problems)
        + "\n"
        + _HOW_TO_FIX
    )


def test_the_completion_methods_are_read_off_the_provider() -> None:
    assert {"acomplete", "astream"} <= COMPLETION_METHODS


def test_every_provider_call_outside_the_providers_reaches_the_status_file() -> None:
    problems = unmarked_calls(_scan())
    assert not problems, _report(problems)


def test_each_conversation_call_is_marked_where_it_is_made() -> None:
    scan = _scan()
    for key, what in CONVERSATION.items():
        calls = [use.line for use, _ in scan.provider if use.fn == key]
        assert len(calls) == 1, f"{key} ({what}) holds {len(calls)} provider calls, not one"
        assert any(line < calls[0] for line in scan.marks.get(key, [])), (
            f"{key} ({what}) makes its call without set_model_call_start before it"
        )


def test_every_library_only_entry_still_holds() -> None:
    scan = _scan()
    for key, (reason, option) in LIBRARY_ONLY.items():
        assert key in scan.fns, f"LIBRARY_ONLY names {key}, which no longer exists"
        if option is None:
            uses = [f"{u.rel}:{u.line}" for u in scan.uses_of(scan.fns[key])]
            assert not uses, f"{key} ({reason}) is used in src/ at {uses}: mark it instead"
        else:
            files = scan.option_files.get(option, set())
            assert _FACADE in files, f"{key} ({reason}): Agent no longer takes {option!r}"
            named = files - {_FACADE}
            assert not named, f"{key} ({reason}): {option!r} is named in {sorted(named)}"


def test_an_unmarked_critic_is_caught_and_the_report_says_what_to_do() -> None:
    # The positive control: the completion critic with its mark taken off, in memory.
    rel = "zakcode/agent/loop.py"
    text = (SRC / rel).read_text(encoding="utf-8")
    mark = '_status_writer().side_call(self.session.id, "critic", model=self.provider.model_id())'
    assert text.count(mark) == 1  # the control edits the real marker, not a copy of the test
    problems = unmarked_calls(Scan(SRC, {rel: text.replace(mark, "contextlib.nullcontext()")}))
    assert [p for p in problems if "AgentLoop._completion_critic" in p and "binary_judge" in p]
    report = _report(problems)
    assert "side_call(session_id" in report and "LIBRARY_ONLY" in report
