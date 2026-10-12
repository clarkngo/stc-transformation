"""
A tiny tracer with the same shape as Langfuse (HOS 4): a trace per
request, spans inside it, and scores attached to it. It records in
memory so the tests run offline. You don't need to change this file.

    trace = obs.start_trace("substitute_order", order_id="O-501")
    with trace.span("lookup_stock") as span:
        out = feature.lookup_stock(order)
        span.set(out_of_stock=out)
    trace.score("substitution_quality", 1.0, comment="milk-whole")
"""

from contextlib import contextmanager

TRACES: list["Trace"] = []


class Span:
    def __init__(self, name: str, attrs: dict):
        self.name = name
        self.attrs = dict(attrs)

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)

    def __repr__(self) -> str:
        return f"Span({self.name!r}, {self.attrs})"


class Trace:
    def __init__(self, name: str, attrs: dict):
        self.name = name
        self.attrs = dict(attrs)
        self.spans: list[Span] = []
        self.scores: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        s = Span(name, attrs)
        self.spans.append(s)
        yield s

    def score(self, name: str, value: float, comment: str = "") -> None:
        self.scores.append({"name": name, "value": value, "comment": comment})

    def spans_named(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]


def start_trace(name: str, **attrs) -> Trace:
    t = Trace(name, attrs)
    TRACES.append(t)
    return t


def reset() -> None:
    TRACES.clear()
