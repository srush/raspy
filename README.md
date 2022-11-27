# raspy

This repo is a reimplementation for [RASP](https://github.com/tech-srl/RASP) as really neat language for thinking about transformer calculations. 

This version is done completely in Python as an [embedded domain specific language](http://wiki.c2.com/?EmbeddedDomainSpecificLanguage).

It mostly runs the RASP code as originally specified: 

```python
def frac_prevs(sop, val):
    prevs = select(indices, indices, le)
    return aggregate(prevs, (sop == val), name="count" + val)


def pair_balance(open, close):
    opens = frac_prevs(tokens, open)
    closes = frac_prevs(tokens, close)
    return opens - closes


bal1 = pair_balance("(", ")")
bal2 = pair_balance("{", "}")

# Check for too many closes
negative = (bal1 < 0) | (bal2 < 0)
had_neg = aggregate(select_all, negative, name="neg") > 0

# Check for match
select_last = select(indices, length - 1, eq)
end_0 = aggregate(select_last, (bal1 == 0) & (bal2 == 0), name="check")

shuffle_dyck2 = end_0 & ~(had_neg)
```

Here is a repo to play around with it:
https://colab.research.google.com/github/srush/raspy/blob/main/RASPy.ipynb
