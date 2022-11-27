from .rasp import (
    Selector,
    SOp,
    SOpLike,
    aggregate,
    eq,
    identity,
    indices,
    lt,
    select,
    wrap,
)

tokens = identity
select_all = select(1, 1, eq)
length = 1 / aggregate(select_all, indices == 0)
reverse = aggregate(select(indices, length - indices - 1, eq), tokens)
same_token = select(tokens, tokens, eq)


def select_eq(k: SOpLike, q: SOpLike) -> Selector:
    return select(k, q, eq)


def where(pred: SOpLike, t: SOpLike, f: SOpLike) -> SOp:
    return wrap(f) * ~wrap(pred) + wrap(t) * wrap(pred)


def selector_width(sel: Selector, assume_bos: bool = False) -> SOp:
    light0 = indices == 0
    or0 = sel | select_eq(indices, 0)
    and0 = sel & select_eq(indices, 0)
    or0_0_frac = aggregate(or0, light0)
    or0_width = 1 / or0_0_frac
    and0_width = aggregate(and0, light0)

    bos_res = or0_width - 1
    nobos_res = bos_res + and0_width
    return bos_res if assume_bos else nobos_res


def has_prev(seq: SOpLike) -> SOp:
    prev_copy = select(seq, seq, eq) and select(indices, indices, lt)
    return aggregate(prev_copy, 1) > 0


def histf(seq: SOpLike, assume_bos: bool = False) -> SOp:
    return selector_width(same_token, assume_bos)


def sort(vals: SOp, keys: SOp, assume_bos: bool = False) -> SOp:
    smaller = select(keys, keys, lt) | (
        select(keys, keys, eq) & select(indices, indices, lt)
    )
    num_smaller = selector_width(smaller, assume_bos=assume_bos)
    if not assume_bos:
        target_pos = num_smaller
    else:
        target_pos = where(indices == 0, 0, (num_smaller + 1))
    sel_new = select(target_pos, indices, eq)
    return aggregate(sel_new, vals)
