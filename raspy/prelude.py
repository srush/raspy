from .rasp import *

tokens = identity
select_all = select(1, 1, eq)
length = 1 / aggregate(select_all, indices==0)
reverse = aggregate(select(indices, length - indices - 1, eq), tokens)
same_token = select(tokens, tokens, eq)
select_eq = lambda k, q: select(k, q, eq)

def selector_width(sel, assume_bos=False):
    light0 = (indices == 0)
    or0 = sel | select_eq(indices, 0)
    and0 = sel & select_eq(indices, 0)
    or0_0_frac = aggregate(or0, light0)
    or0_width = 1 / or0_0_frac
    and0_width = aggregate(and0, light0, 0)
    
    bos_res = or0_width - 1
    nobos_res = bos_res + and0_width
    return bos_res if assume_bos else nobos_res

def has_prev(seq):
    prev_copy = select(seq, seq, eq) and select(indices, indices, lt)
    return aggregate(prev_copy, 1, 0) > 0

def histf(seq, assume_bos=False):
    return selector_width(same_token, assume_bos)

def sort(vals, keys, assume_bos=False):
    smaller = select(keys, keys, lt) | (select(keys, keys, eq)  & select(indices, indices, lt))
    num_smaller = selector_width(smaller, assume_bos=assume_bos)
    if not assume_bos:
        target_pos = num_smaller
    else:
        target_pos = where(indices == 0, 0, (num_smaller + 1))
    sel_new = select(target_pos, indices, eq)
    return aggregate(sel_new, vals)
