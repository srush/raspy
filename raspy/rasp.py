from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

Raw = int
SOpLike = Union["SOp", List[Raw], str, int, float]
EXAMPLE = "hello"


class Seq:
    def __init__(
        self,
        val: List[Raw],
        layer: int = 0,
        bind: Optional[List[Raw]] = None,
        sels: List[BoundSel] = [],
    ):
        self.val = val
        self.layer = layer
        if bind is None:
            self.bind = val
        else:
            self.bind = bind
        self.sels = list(set(sels))

    def toseq(self) -> List[Raw]:
        return self.val

    def __repr__(self) -> str:
        return repr(self.val)

    def __hash__(self) -> int:
        return hash(tuple(self.val))


def show(e: Any) -> str:
    if isinstance(e, bool):
        return str(int(e))
    else:
        return str(e)


@dataclass
class Sel:
    kqs: List[Tuple[Seq, Seq]]
    layer: int
    val: List[List[bool]]
    bind: List[Raw]

    def __repr__(self) -> str:
        k, _ = [kq.val for kq in self.kqs[0]]
        mat = self.val
        out = f"Selector Layer {self.layer}\n"
        out += (
            "        " + " ".join([show(x % 10) for x in range(len(self.bind))]) + "\n"
        )
        out += "        |" + " ".join([show(x) for x in self.bind]) + "|\n"
        for _, q in self.kqs:
            out += "        " + " ".join([show(x) for x in q.val]) + "\n"
        out += "        |" + "-".join(["-" for x in self.bind]) + "|\n"
        for e, row, t, i in zip(k, mat, self.bind, range(len(self.bind))):
            out += (
                show(i % 10)
                + " "
                + show(t)
                + " "
                + show(e)
                + " | "
                + " ".join(["1" if v else " " for v in row])
                + "\n"
            )
        return out


total_sel = 0


@dataclass
class BoundSel:
    sel: Sel
    v: Seq
    result: List[Raw]
    selid: int
    layer: int
    width: bool = False
    _name: str = ""

    def name(self, name: str) -> BoundSel:
        if not self._name:
            self._name = name
        return self

    def __hash__(self) -> int:
        return self.selid

    def __repr__(self) -> str:
        sel = self.sel
        k, _ = [kq.val for kq in sel.kqs[0]]
        mat = sel.val
        out = f"Selector Layer {sel.layer}\n"
        out += (
            "        " + " ".join([show(x % 10) for x in range(len(sel.bind))]) + "\n"
        )
        out += "        " + " ".join([show(x) for x in sel.bind]) + "\n"
        for _, q in sel.kqs:
            out += "        " + " ".join([show(x) for x in q.val]) + "\n"
        out += "      |" + "-".join(["-" for x in sel.bind]) + "--|\n"
        for e, row, t, i, v2 in zip(k, mat, sel.bind, range(len(sel.bind)), self.v.val):
            out += (
                show(i % 10)
                + " "
                + show(t)
                + " "
                + show(e)
                + " | "
                + " ".join(["1" if v else " " for v in row])
                + " | "
                + show(v2)
                + " \n"
            )
        out += "      |" + "-".join(["-" for x in sel.bind]) + "--|\n"

        # if any([isinstance(r, float) for r in self.result]):
        #     result = [
        #         mean2([self.v.val[j] for j in range(len(self.v.val)) if sel.val[j][i]])
        #         for i in range(len(sel.val))
        #     ]
        #     out += "        " + " ".join([show(x[0]) for x in result]) + "\n"
        #     out += "        " + " ".join([show("-") for x in result]) + "\n"
        #     out += "        " + " ".join([show(x[1]) for x in result]) + "\n"
        # else:
        out += "        " + " ".join([show(x) for x in self.result]) + "\n"
        return out


class SOp(Number):
    def __init__(self, f: Callable[[Seq], Seq]):
        self.f = f
        self.cache: Dict[Seq, Seq] = {}

    @staticmethod
    def zip(f, *ss: SOp) -> SOp:  # type: ignore
        def fn(s: Seq) -> Seq:
            seq1 = [s1(s) for s1 in ss]

            return Seq(
                [f(*vs) for vs in zip(*[s1.val for s1 in seq1])],
                reduce(max, [s1.layer for s1 in seq1]),
                s.val,
                [sel for s1 in seq1 for sel in s1.sels],
            )

        return SOp(fn)

    def name(self, name: str) -> SOp:
        def f(x: Seq) -> Seq:
            seq = self.f(x)
            seq.sels = [sel.name(name) for sel in seq.sels]
            return seq

        return SOp(f)

    def map(self, f: Callable[[Raw], Raw]) -> SOp:
        return SOp.zip(f, self)

    def __call__(self, inplike: Any) -> Seq:
        inp = inplike
        if not isinstance(inplike, Seq):
            inp = Seq(inp)
        if inp not in self.cache:
            self.cache[inp] = self.f(inp)
        return self.cache[inp]

    def _op(self, f, *other) -> SOp:  # type: ignore
        return SOp.zip(f, self, *map(wrap, other))

    def round(self) -> SOp:
        return self.map(lambda x: int(round(x)))

    def has(self, v: Any) -> SOp:
        return self.map(lambda x: x in v)

    def __repr__(self) -> str:
        return repr(self(EXAMPLE))

    def __rtruediv__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda b, a: a / b, self, wrap(x))

    def __hash__(self) -> int:
        return 0


def numeric_ops(cls) -> None:  # type: ignore
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
        "mod",
    ]:
        n = f"__{op}__"

        def _temp_(self, *other, n=n):  # type: ignore
            return self._op(getattr(operator, n), *other)

        x = _temp_
        x.__name__ = f"__{op}__"
        setattr(cls, f"__{op}__", x)


numeric_ops(SOp)


def raw(x: List[Raw]) -> SOp:
    return SOp(lambda _: Seq(x))


def wrap(x: SOpLike) -> SOp:
    if isinstance(x, SOp):
        return x
    if (
        isinstance(x, float)
        or isinstance(x, int)
        or isinstance(x, str)
    ):
        return repeat(x)  # type: ignore
    return raw(x)  # type: ignore


def repeat(y: Raw) -> SOp:
    return SOp(lambda x: Seq([y for _ in range(len(x.val))]))


indices = SOp(lambda x: Seq([i for i in range(len(x.val))], 0, x.val))
identity = SOp(lambda x: x)


class Selector(Number):
    def __init__(self, s: Callable[[Seq], Sel]):
        self.s = s

    @staticmethod
    def zip(fn, *aa: Selector) -> Selector:  # type: ignore
        def ret(x: Seq) -> Sel:
            av = [a.s(x) for a in aa]
            return Sel(
                [kq for a in av for kq in a.kqs],
                reduce(max, [a.layer for a in av]),
                [[fn(*ax2) for ax2 in zip(*ax)] for ax in zip(*[al.val for al in av])],
                x.val,
            )

        return Selector(ret)

    def map(self, fn: Callable[[bool], bool]) -> Selector:
        return Selector.zip(fn)

    def __call__(self, inplike: List[Raw]) -> Sel:
        return self.s(Seq(inplike))

    def __repr__(self) -> str:
        return repr(self(EXAMPLE))  # type: ignore

    def _op(self, op, *other: Selector) -> Selector:  # type: ignore
        return Selector.zip(op, self, *other)

    def value(self, val: SOpLike, default: Raw = 0) -> SOp:
        return aggregate(self, val, default=default)

    def __hash__(self) -> int:
        return 0


def select(
    keylike: SOpLike, querylike: SOpLike, predicate: Callable[[Raw, Raw], bool]
) -> Selector:
    query = wrap(querylike)
    key = wrap(keylike)

    def ret(x: Seq) -> Sel:
        q, k = query(x), key(x)
        return Sel(
            [(k, q)],
            max(k.layer, q.layer),
            [
                [predicate(k.val[j], q.val[i]) for i in range(len(q.val))]
                for j in range(len(k.val))
            ],
            x.val,
        )

    return Selector(ret)


def mean(x: List[int], default: Raw = 0) -> int:
    if len(x) == 0:
        return default
    if len(x) == 1:
        return x[0]
    return sum(x)  # / len(x)


# def mean2(x: List[float], default: Raw = 0) -> float:
#     if len(x) == 0:
#         return default, 1
#     if len(x) == 1:
#         return x[0], 1
#     return sum(x), len(x)


def aggregate(sel: Selector, vallike: SOpLike, default: Any = 0, name: str = "") -> SOp:
    val = wrap(vallike)

    def fn(x: Seq) -> Seq:
        v = val(x)
        s = sel.s(x)
        result = [
            mean(
                [v.val[j] for j in range(len(v.val)) if s.val[j][i]],
                default=default,
            )
            for i in range(len(s.val))
        ]
        global total_sel
        total_sel += 1
        lay = max(s.layer + 1, v.layer + 1)
        return Seq(
            result,
            layer=lay,
            bind=x.val,
            sels=v.sels
            + [BoundSel(s, v, result, total_sel, lay)]
            + [sel for k, q in s.kqs for seq in [k, q] for sel in seq.sels],
        )

    return SOp(fn)


class Key(Number):
    def __init__(self, sop: SOp):
        self.sop = sop

    def _op(self, f, other: Query) -> Selector:  # type: ignore
        return select(self.sop, other.sop, f)

    def __hash__(self) -> int:
        return 0


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
            f"query(out)({repr(run)})=\n"
            + " | \n".join(map(str, self.sop(run).toseq()))
            + " |"
        )


def query(x: SOpLike) -> Query:
    return Query(x)


def where(pred: SOpLike, t: SOpLike, f: SOpLike) -> SOp:
    return SOp.zip(lambda p, t, f: t if p else f, wrap(pred), wrap(t), wrap(f))


tokens = identity
