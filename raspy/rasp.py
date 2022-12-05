from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, List, Optional, Tuple, Union

Raw = float
SOpLike = Union["SOp", List[Raw], str, int, float]
Att = List[List[bool]]


def numeric_ops(cls) -> None:
    for op in [
        "le",
        "lt",
        "gt",
        "ge",
        "eq",
        "ne",
        "truediv",
        "add",
        "sub",
        "and",
        "or",
        "mul",
        "invert",
    ]:
        n = f"__{op}__"
        x = lambda self, *other, n=n: self._op(getattr(operator, n), *other)
        x.__name__ = f"__{op}__"
        setattr(cls, f"__{op}__", x)


# History
EXAMPLE = "hello"


def set_example(example: Any) -> None:
    global EXAMPLE
    EXAMPLE = example


@dataclass
class Hist:
    layer: int
    prev: List[Hist]
    attentions: Optional[Tuple[Att, str]] = None


empty = Hist(0, [], None)


def merge(*hists: Hist) -> Hist:
    return Hist(
        layer=reduce(max, [h.layer for h in hists]), prev=hists, attentions=None
    )


# Raw values.
@dataclass
class Seq:
    val: List[Raw]
    hist: Hist = empty

    def toseq(self) -> List[Raw]:
        return self.val

    def __repr__(self) -> str:
        return repr(self.val)


@dataclass
class Sel:
    val: Att
    hist: Hist = empty

    def __repr__(self):
        run = self.val
        mat = self.val
        out = "    " + " ".join([str(x) for x in run]) + "\n"
        out += "    " + "-".join(["-" for x in run]) + "\n"
        for e, row in zip(run, mat):
            out += str(e) + " | " + " ".join(["1" if v else " " for v in row]) + "\n"
        return f"out({repr(run)})= \n" + out


class SOp:
    def __init__(self, f: Callable[[Seq], Seq]):
        self.f = f
        self.cache = {}
        
    @staticmethod
    def zip(f, *ss: SOp) -> SOp:
        def fn(s: Seq) -> Seq:
            seq1 = [s1(s) for s1 in ss]

            return Seq(
                [f(*vs) for vs in zip(*[s1.val for s1 in seq1])],
                merge(*[s1.hist for s1 in seq1]),
            )

        return SOp(fn)

    def map(self, f: Callable[[Raw], Raw]) -> SOp:
        return SOp.zip(f, self)

    def __call__(self, inplike) -> Seq:
        inp = inplike
        if not isinstance(inplike, Seq):
            inp = Seq(inp)
        key = tuple(inp.val)
        if key not in self.cache:
            self.cache[key] = self.f(inp)
        return self.cache[key]

    def _op(self, f, *other) -> SOp:
        return SOp.zip(f, self, *map(wrap, other))

    def has(self, v: Any) -> SOp:
        return self.map(lambda x: x in v)

    def cos(self) -> SOp:
        return self.map(math.cos)

    def sin(self) -> SOp:
        return self.map(math.sin)

    def __repr__(self) -> str:
        return repr(self(EXAMPLE))

    def __rtruediv__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda b, a: a / b, self, wrap(x))

    def totree(self) -> None:
        def collect(history: Hist) -> None:
            queue = [history]
            seen = {}
            while queue:
                queue.sort(key=lambda x: x.layer, reverse=True)
                cur = queue[0]
                queue = queue[1:]
                if cur.attentions is not None:
                    if (cur.attentions[1], cur.layer) in seen:
                        continue
                    print()
                    print(f"Layer {cur.layer} Head: {cur.attentions[1]} \n")
                    out = "    " + " ".join([str(x) for x in EXAMPLE]) + "\n"
                    out += "    " + "-".join(["-" for x in EXAMPLE]) + "\n"
                    for e, row in zip(EXAMPLE, cur.attentions[0]):
                        out += (
                            str(e)
                            + " | "
                            + " ".join(["1" if v else " " for v in row])
                            + "\n"
                        )
                    print(out)
                    seen[cur.attentions[1], cur.layer] = True
                queue += cur.prev

        history = self(EXAMPLE).hist
        print("Number of layers", history.layer)
        collect(history)


numeric_ops(SOp)


def raw(x: List[Raw]) -> SOp:
    return SOp(lambda _: Seq(x))


def wrap(x: SOpLike) -> SOp:
    if isinstance(x, SOp):
        return x
    if (
        isinstance(x, float)
        or isinstance(x, int)
        or (isinstance(x, str) and len(x) == 1)
    ):
        return repeat(x)  # type: ignore
    return raw(x)  # type: ignore


def repeat(y: Raw) -> SOp:
    return SOp(lambda x: Seq([y for _ in range(len(x.val))]))


indices = SOp(lambda x: Seq([i for i in range(len(x.val))]))
identity = SOp(lambda x: x)


class Selector:
    def __init__(self, s: Callable[[Seq], Sel]):
        self.s = s
        self.cache = {}
        
    @staticmethod
    def zip(fn: Callable[[bool], bool], *aa: Selector) -> Selector:
        def ret(x: Seq) -> Sel:
            av = [a.s(x) for a in aa]
            return Sel(
                [[fn(*ax2) for ax2 in zip(*ax)]
                 for ax in zip(*[al.val for al in av])],
                merge(*[a.hist for a in av]),
            )

        return Selector(ret)

    def map(self, fn: Callable[[bool], bool]) -> Selector:
        return Selector.zip(fn)

    def __call__(self, inplike: SOpLike) -> Sel:
        return self.f(Seq(inplike))

    def __repr__(self) -> str:
        return repr(self(EXAMPLE))

    def _op(self, op, *other: Selector) -> Selector:
        return Selector.zip(op, self, *other)

    def value(self, val: SOpLike, default: Raw = 0) -> SOp:
        return aggregate(self, val, default=default)


def select(
    keylike: SOpLike, querylike: SOpLike, predicate: Callable[[Raw, Raw], bool]
) -> Selector:
    query = wrap(querylike)
    key = wrap(keylike)

    def ret(x: Seq) -> Sel:
        q, k = query(x), key(x)
        return Sel(
            [
                [predicate(k.val[j], q.val[i]) for i in range(len(q.val))]
                for j in range(len(k.val))
            ],
            merge(q.hist, k.hist),
        )

    return Selector(ret)


def mean(x: List[float], default: raw = 0) -> float:
    if len(x) == 0:
        return default
    if len(x) == 1:
        return x[0]
    return sum(x) / len(x)


def aggregate(sel: Selector, vallike: SOpLike, default=0, name: str = "") -> SOp:
    val = wrap(vallike)

    def fn(x: Seq) -> Seq:
        v = val(x)
        s = sel.s(x)
        return Seq(
            [
                mean(
                    [v.val[j] for j in range(len(v.val)) if s.val[j][i]],
                    default=default,
                )
                for i in range(len(s.val))
            ],
            Hist(
                layer=v.hist.layer + 1,
                prev=v.hist.prev,
                attentions=(s.val, name),
            ),
        )

    return SOp(fn)


class Key:
    def __init__(self, sop: SOp):
        self.sop = sop

    def _op(self, f: Callable[[Raw, Raw], Raw], other: Query) -> Selector:
        return select(self.sop, other.sop, f)


numeric_ops(Key)
numeric_ops(Selector)


def key(x: SOpLike) -> Key:
    return Key(wrap(x))


class Query:
    def __init__(self, sop: SOpLike):
        self.sop = wrap(sop)

    def __repr__(self) -> str:
        run = EXAMPLE
        return (
            f"query(out)({repr(run)})=\n" + " | \n".join(self.sop(run).toseq()) + " |"
        )


def query(x: SOpLike) -> Query:
    return Query(x)

def where(pred: SOpLike, t: SOpLike, f: SOpLike) -> SOp:
    return SOp.zip(lambda p, t, f: t if p else f, wrap(pred), wrap(t), wrap(f))

