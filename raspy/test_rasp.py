from raspy import *

def check(x, y):
    for a, b in zip(x.toseq(), y):
        assert a == b

def check_sel(x, y):
    for a, b in zip(x.tomat(), y):
        for ax, bx in zip(a, b):
            assert ax == bx

def test_raw():
    check(raw([1,2,3,4]), [1,2,3,4])

def test_sum():
    orig1 = [1,2,3,4]
    orig2 = [2,3,4,5]
    check(raw(orig1) + raw(orig2), [3, 5, 7, 9])


def test_neg():
    check(-raw([1,2,3,4]), [-1,-2,-3,-4])

        
def test_select():
    orig1 = [1,1,2,2]
    orig2 = [1,3,2,1]
    sel = select(raw(orig2), raw(orig1), eq)
    s = sel.tomat()
    for i in range(4):
        for j in range(4): 
            assert s[i][j] == (orig1[i] == orig2[j])

    check_sel(sel, [[1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0]])

        
def test_select():
    orig1 = [1,1,2,2]
    orig2 = [1,3,2,1]
    sel = select(orig2, orig1, eq)
    sel = sel & select_all(orig1)
    check_sel(sel, [[1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0]])
    sel = sel | select_all(orig1)
    check_sel(sel, [[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]])

    
def test_agg():
    orig1 = [1,1,2,2]
    orig2 = [1,3,2,1]
    sel = select(raw(orig1), raw(orig2), eq)
    check(aggregate(sel, raw([1,2,3,4])), [1.5, 0, 3.5, 1.5])


def test_index():
    check(indices("hi"), [0,1])

def test_id():
    check(identity("hi"), ["h", "i"])

def test_length():
    check(length("hi"), [2, 2])


def test_example():
    s = select([0, 1, 2], [1, 2, 3], lt)
    check(aggregate(s, [10, 20, 30]), [10, 15, 20])

def test_reverse():
    flip = select(indices, length - indices - 1, eq)
    reverse = aggregate(flip, identity)
    check(reverse("hey"), ["y", "e", "h"])




def test_shuffle_dyck():
    def frac_prevs(sop, val):
        prevs = select(indices, indices, le)
        return aggregate(prevs, (sop == val), "count")

    def pair_balance(open, close):
        opens = frac_prevs(tokens, open)
        closes = frac_prevs(tokens, close)
        return opens - closes

    bal1 = pair_balance("(", ")")
    bal2 = pair_balance("{", "}")

    # Check for too many closes
    negative = (bal1 < 0) | (bal2 < 0)
    had_neg = aggregate(select_all, negative, "neg") > 0

    # Check for match
    select_last = select(indices, length - 1, eq)
    end_0 = aggregate(select_last, (bal1 == 0) & (bal2 == 0), "check")
    
    shuffle_dyck2 = end_0 & ~(had_neg)
    print(shuffle_dyck2("(({{a))b}}").totree())
    assert shuffle_dyck2("(({{a))b}}").toseq()[-1]
    assert not shuffle_dyck2("(({{{a))b}}").toseq()[-1]
    

def test_reverse():
    check(reverse([1,2,3]), [3,2,1])

def test_in():
    check(raw([1,2,5]).has([1,2,3]), [True, True, False])

def test_selector_width():
    check(selector_width(same_token)("hello"), [1,1,2,2,1])

def test_sort():
    check(sort(tokens, tokens)([3, 1, 2, 4]), [1,2,3,4]) 
