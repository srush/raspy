from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

Pos = int
Raw = float
SOpLike = Union["SOp", List[Raw], str, int, float]
eq = lambda a, b: a == b  # noqa: E731
neq = lambda a, b: a != b  # noqa: E731
lt = lambda a, b: a < b  # noqa: E731
le = lambda a, b: a <= b  # noqa: E731
gt = lambda a, b: a > b  # noqa: E731
ge = lambda a, b: a >= b  # noqa: E731
lor = lambda a, b: a or b  # noqa: E731
land = lambda a, b: a and b  # noqa: E731
Att = List[List[bool]]


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


def merge(hist1: Hist, hist2: Hist) -> Hist:
    return Hist(
        layer=max(hist1.layer, hist2.layer), prev=[hist1, hist2], attentions=None
    )


# Raw values.
@dataclass
class Seq:
    val: List[Raw]
    hist: Hist = empty


@dataclass
class Sel:
    val: Att
    hist: Hist = empty


class SOp:
    def __init__(self, f: Callable[[Seq], Seq]):
        self.f = f

    def map(self, f: Callable[[Raw], Raw]) -> SOp:
        def fn(s: Seq) -> Seq:
            seq = self.f(s)
            return Seq([f(v) for v in seq.val], seq.hist)

        return SOp(fn)

    @staticmethod
    def zip(f: Callable[[Raw, Raw], Raw], s1: SOp, s2: SOp) -> SOp:
        def fn(s: Seq) -> Seq:
            seq1 = s1.f(s)
            seq2 = s2.f(s)
            return Seq(
                [f(a, b) for a, b in zip(seq1.val, seq2.val)],
                merge(seq1.hist, seq2.hist),
            )

        return SOp(fn)

    def __call__(self, inplike: SOpLike) -> SOp:
        inp = wrap(inplike)

        def fn(s: Seq) -> Seq:
            seq1 = inp.f(s)
            return self.f(seq1)

        return SOp(fn)

    def __eq__(self, x: SOpLike) -> SOp:  # type: ignore
        return SOp.zip(eq, self, wrap(x))

    def __neq__(self, x: SOpLike) -> SOp:  # type: ignore
        return SOp.zip(neq, self, wrap(x))

    def __le__(self, x: SOpLike) -> SOp:
        return SOp.zip(le, self, wrap(x))

    def __lt__(self, x: SOpLike) -> SOp:
        return SOp.zip(lt, self, wrap(x))

    def __ge__(self, x: SOpLike) -> SOp:
        return SOp.zip(ge, self, wrap(x))

    def __gt__(self, x: SOpLike) -> SOp:
        return SOp.zip(gt, self, wrap(x))

    def __add__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda a, b: a + b, self, wrap(x))

    def __sub__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda a, b: a - b, self, wrap(x))

    def __neg__(self) -> SOp:
        return self.map(lambda a: -a)

    def __mul__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda a, b: a * b, self, wrap(x))

    def __rmul__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda b, a: a * b, self, wrap(x))

    def __truediv__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda a, b: a / b, self, wrap(x))

    def __rtruediv__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda b, a: a / b, self, wrap(x))

    def __invert__(self) -> SOp:
        return self.map(lambda a: not a)

    def __and__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda a, b: a and b, self, wrap(x))

    def __or__(self, x: SOpLike) -> SOp:
        return SOp.zip(lambda a, b: a or b, self, wrap(x))

    def has(self, v: Any) -> SOp:
        return self.map(lambda x: x in v)

    def cos(self) -> SOp:
        return self.map(math.cos)

    def sin(self) -> SOp:
        return self.map(math.sin)

    def toseq(self) -> List[Raw]:
        return self.f(Seq([])).val

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
                        out += str(e) + " | " + " ".join(["1" if v else " "  for v in row]) + "\n"
                    print(out)
                    seen[cur.attentions[1], cur.layer] = True
                queue += cur.prev

        history = self(EXAMPLE).f(Seq([])).hist
        print("Number of layers", history.layer)
        collect(history)

    def __repr__(self) -> str:
        global EXAMPLE
        return f"out({EXAMPLE})" + repr(self(EXAMPLE).toseq())


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

    def tomat(self) -> List[List[bool]]:
        s = self.s(Seq([]))
        return [[s.val[i][j] for i in range(len(s.val))] for j in range(len(s.val))]

    @staticmethod
    def zip(fn: Callable[[bool, bool], bool], a: Selector, b: Selector) -> Selector:
        def ret(x: Seq) -> Sel:
            av: Sel = a.s(x)
            bv: Sel = b.s(x)
            return Sel(
                [
                    [fn(ax, bx) for ax, bx in zip(al, bl)]
                    for al, bl in zip(av.val, bv.val)
                ],
                merge(av.hist, bv.hist),
            )

        return Selector(ret)

    def map(self, fn: Callable[[bool], bool]) -> Selector:
        def ret(x: Seq) -> Sel:
            sv = self.s(x)
            return Sel([[fn(ax) for ax in al] for al in sv.val], sv.hist)

        return Selector(ret)

    def __call__(self, inplike: SOpLike) -> Selector:
        inp = wrap(inplike)

        def ret(x: Seq) -> Sel:
            seq = inp.f(x)
            return self.s(seq)

        return Selector(ret)

    def __repr__(self) -> str:
        global EXAMPLE
        mat = self(EXAMPLE).tomat()
        out = "    " + " ".join([str(x) for x in EXAMPLE]) + "\n"
        out += "    " + "-".join(["-" for x in EXAMPLE]) + "\n"
        for e, row in zip(EXAMPLE, mat):
            out += str(e) + " | " + " ".join(["1" if v else " "  for v in row]) + "\n"
        return f"out({EXAMPLE})= \n" + out

    def __invert__(self) -> Selector:
        return self.map(lambda a: not a)

    def __and__(self, x: Selector) -> Selector:
        return Selector.zip(lambda a, b: a and b, self, x)

    def __or__(self, x: Selector) -> Selector:
        return Selector.zip(lambda a, b: a or b, self, x)


def select(
    keylike: SOpLike, querylike: SOpLike, predicate: Callable[[Raw, Raw], bool]
) -> Selector:
    query = wrap(querylike)
    key = wrap(keylike)

    def ret(x: Seq) -> Sel:
        q, k = query.f(x), key.f(x)
        return Sel(
            [
                [predicate(k.val[j], q.val[i]) for i in range(len(q.val))]
                for j in range(len(k.val))
            ],
            merge(q.hist, k.hist),
        )

    return Selector(ret)


def mean(x: List[float]) -> float:
    if len(x) == 0:
        return 0
    if len(x) == 1:
        return x[0]
    return sum(x) / len(x)


def aggregate(sel: Selector, vallike: SOpLike, name: str = "") -> SOp:
    val = wrap(vallike)

    def fn(x: Seq) -> Seq:
        v = val.f(x)
        s = sel.s(x)
        return Seq(
            [
                mean([v.val[j] for j in range(len(v.val)) if s.val[j][i]])
                for i in range(len(s.val))
            ],
            Hist(
                layer=v.hist.layer + 1,
                prev=v.hist.prev,
                attentions=(s.val, name),
            ),
        )

    return SOp(fn)
