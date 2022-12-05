from .rasp import (
    Selector,
    SOp,
    SOpLike,
    aggregate,
    identity,
    indices,
    key,
    query,
    select,
    wrap,
)

tokens = identity
select_all = key(1) == query(1)
length = 1 / aggregate(select_all, indices == 0)


def select_eq(k: SOpLike, q: SOpLike) -> Selector:
    return key(k) == query(q)


reverse = aggregate(select_eq(indices, length - indices - 1), tokens)
same_token = key(tokens) == query(tokens)


def selector_width(sel: Selector, assume_bos: bool = False) -> SOp:
    light0 = indices == 0
    or0 = sel | select_eq(indices, 0)
    or0_0_frac = aggregate(or0, light0)
    or0_width = 1 / or0_0_frac

    and0 = sel & select_eq(indices, 0)
    and0_width = aggregate(and0, light0)

    bos_res = or0_width - 1
    nobos_res = bos_res + and0_width
    return bos_res if assume_bos else nobos_res


def has_prev(seq: SOpLike) -> SOp:
    prev_copy = select_eq(seq, seq) and (key(indices) < query(indices))
    return aggregate(prev_copy, 1) > 0


def histf(seq: SOpLike, assume_bos: bool = False) -> SOp:
    return selector_width(same_token, assume_bos)


def sort(vals: SOp, keys: SOp, assume_bos: bool = False) -> SOp:
    smaller = (key(keys) < query(keys)) | (
        (key(keys) == query(keys)) & (key(indices) < query(indices))
    )
    num_smaller = selector_width(smaller, assume_bos=assume_bos)
    if not assume_bos:
        target_pos = num_smaller
    else:
        target_pos = where(indices == 0, 0, (num_smaller + 1))
    sel_new = select_eq(target_pos, indices)
    return aggregate(sel_new, vals)
