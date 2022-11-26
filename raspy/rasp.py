from __future__ import annotations
from typing import Callable, List, Tuple, Union
from dataclasses import dataclass
import math

Pos = int
Raw = Union[int, str]
SOpLike = Union["SOp", List[Raw], str]

class Seq:
    def __init__(self, x: List[Raw], att=[]):
        self.val = x
        self.att = att
        
eq = lambda a, b: a ==b
neq = lambda a, b: a != b
lt = lambda a, b: a < b
le = lambda a, b: a <= b
gt = lambda a, b: a > b
ge = lambda a, b: a >= b

Att = List[List[bool]]

@dataclass
class Par:
    a: Union[Par, Sequence]
    b: Union[Par, Sequence]

@dataclass
class Sequence:
    a: Union[Par, Sequence]
    b: Union[Par, Sequence]


    
class SOp:
    def __init__(self, f: Callable[[List[Raw]], List[Raw]]):
        self.f = f

    def map(s1: SOp, f: Callable[[Raw], Raw]) -> SOp:
        def fn(x, h):
            seq = s1.f(x, h)
            return Seq([f(v) for v in seq.val],
                       seq.att)
        
        return SOp(fn)

    @staticmethod
    def zip(f: Callable[[Raw], Raw],  s1: SOp, s2: SOp) -> SOp:
        def fn(x, h):
            seq1 = s1.f(x, h)
            seq2 = s2.f(x, h)
            return Seq([f(a, b) for a, b in zip(seq1.val, seq2.val)],
                       Par(seq1.att, seq2.att))
        return SOp(fn)
    
    def __call__(self, inp: SOpLike) -> SOp:
        inp = wrap(inp)
        def fn(x, h):
            seq1 = inp.f(x, h)
            seq2 = self.f(seq1.val, seq1.att)
            return Seq(seq2.val, seq2.att)
        return SOp(fn)

    
    def __eq__(self, x: SOpLike) -> SOp:
        return SOp.zip(eq, self, wrap(x))

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
        return self.f(None, []).val

    def totree(self) -> List[Raw]:
        def collect(att):
            v = []
            if isinstance(att, Par):
                v += collect(att.a)
                return v + collect(att.b)
            if isinstance(att, Sequence):
                v += collect(att.a)
                return v + collect(att.b)
            if isinstance(att, tuple):
                return [att]
            return []
        for l, name in collect(self.f(None, []).att):
            print()
            print(name)
            for r in l:
                print(r)

    def __repr__(self) -> str:
        return repr(self.toseq())

def raw(x:List[Raw]) -> SOp:
    return SOp(lambda _, h: Seq(x))


def wrap(x: SOpLike) -> SOp:
    if isinstance(x, SOp):
        return x
    if isinstance(x, float) or isinstance(x, int) or (isinstance(x, str) and len(x) == 1):
        return repeat(x)
    return raw(x)

def repeat(y:Raw) -> SOp:
    return SOp(lambda x, h: Seq([y for _ in range(len(x))]))

indices = SOp(lambda x, h: Seq([i for i in range(len(x))]))
identity = SOp(lambda x, h: Seq(x))


class Selector:
    def __init__(self, s: Callable[[List[Raw]], List[List[Bool]]]):
        self.s = s

    def tomat(self) -> List[List[bool]]:
        s = self.s(None, [])
        return [[s.val[i][j] for i in range(len(s.val))]
                for j in range(len(s.val))]

    @staticmethod
    def zip(fn, a: Selector, b: Selector):
        def ret(x, h):
            av, bv = a.s(x, h), b.s(x, h)
            return Seq([[fn(ax, bx) for ax, bx in zip(al, bl)]
                        for al, bl in zip(av.val, bv.val)],
                       Par(av.att, bv.att))
        return Selector(ret)

    
    def map(self, fn):
        def ret(x, h):
            sv = self.s(x, h)
            return Seq([[fn(ax) for ax in al] for al in sv.val], sv.att)
        return Selector(ret)
    
    def __call__(self, inp: SOpLike) -> str:
        inp = wrap(inp)
        def ret(x, h):
            seq = inp.f(x, h)
            return self.s(seq.val, seq.att)
        return Selector(ret)

    def __repr__(self) -> str:
        return repr(self.tomat())
    
    def __invert__(self) -> Selector:
        return self.map(lambda a: not a)

    def __and__(self, x: Selector) -> Selector:
        return Selector.zip(lambda a, b: a and b, self, x)

    def __or__(self, x: Selector) -> Selector:
        return Selector.zip(lambda a, b: a or b, self, x)

    
def select(key: SOpLike, query: SOpLike,
           predicate: Callable[[Raw, Raw], bool]) -> Callable[[SOp], Selector]:
    query = wrap(query)
    key = wrap(key)
    def ret(x, h):
        q, k = query.f(x, h), key.f(x, h)
        return Seq([[predicate(k.val[j], q.val[i])
                     for i in range(len(q.val))]
                    for j in range(len(k.val))],
                   Par(q.att, k.att))
    return Selector(ret)

def mean(x):
    if len(x) == 0:
        return 0
    if len(x) == 1:
        return x[0]
    return sum(x) / len(x)

def aggregate(sel: Selector, val: SOpLike, name="") -> SOp:
    val = wrap(val)
    def fn(x, h):
        v = val.f(x, h)
        s = sel.s(x, h)
        return Seq([mean([v.val[j] for j in range(len(v.val))
                          if s.val[j][i]])
                    for i in range(len(s.val))],
                   Sequence(v.att, (s.val, name)))
    return SOp(fn)


def where(pred, t, f):
    return f * ~pred + t * pred
