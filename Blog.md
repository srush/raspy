  
  <link rel="stylesheet" href="custom.css">



# Thinking Like Transformers

- [Paper](https://arxiv.org/pdf/2106.06981.pdf) by Gail Weiss, Yoav Goldberg, Eran Yahav
- Blog by [Sasha Rush](https://rush-nlp.com/) and [Gail Weiss](https://gailweiss.github.io/)
- Library and Interactive Notebook: [srush/raspy](https://github.com/srush/RASPy)

Transformer models are foundational to AI systems. There are now countless explanations of "how transformers work?" in the sense of the architecture diagram at the heart of transformers.





    
![svg](Blog_files/Blog_5_0.svg)
    



However this diagram does not provide any intuition into the computational model of this framework. As researchers become interested in how Transformers work, gaining intuition into their mechanisms becomes increasingly useful.

<a href="https://arxiv.org/pdf/2106.06981.pdf">Thinking like Transformers</a> proposes a computational framework for Transformer-like calculations. The framework uses discrete computation to simulate Transformer computations. The resulting language <a href="https://github.com/tech-srl/RASP">RASP</a> is a programming language where, ideally, every program can compile down to a specific Transformer (indeed, David Lindner and colleagues have recently released a <a href="https://arxiv.org/abs/2301.05062">compiler</a> for a large subset of RASP!).</p>


In this blog post, I reimplemented a variant of RASP in Python ([RASPy](https://github.com/srush/raspy)). The language is roughly compatible with the original version, but with some syntactic changes that I thought were fun. With this language, the author of the work Gail Weiss, provided a challenging set of puzzles to walk through and understand how it works. 

```
#!pip install git+https://github.com/srush/RASPy
```
Before jumping into the language itself, let's look at an example of what coding with Transformers looks like. Here is some code that computes the `flip`, i.e. reversing an input sequence. The code itself uses two Transformer layers to apply attention and mathematical computations to achieve the result.

```
def flip():
    length = (key(1) == query(1)).value(1)
    flip = (key(length - indices - 1) == query(indices)).value(tokens)
    return flip
flip()
```



    
![svg](Blog_files/Blog_11_0.svg)
    



<h2 id="table-of-contents">Table of Contents</h2>
<ul>
<li>Part 1: <a href="#transformers-as-code">Transformers as Code</a></li>
<li>Part 2: <a href="#coding-with-transformers">Coding with Transformers</a></li>
</ul>


## Transformers as Code

Our goal is to define a computational formalism that mimics the expressivity of Transformers. We will go through this process by analogy, describing each language construct next to the aspect of the Transformer it represents. (See the full [paper](https://arxiv.org/pdf/2106.06981.pdf) for the formal language specification).

The core unit of the language is a *sequence operation* that transforms a sequence to another sequence of the same length. I will refer to these throughout as *transforms*.

### Inputs

In a Transformer, the base layer is the input fed to the model. This input usually contains the raw tokens as well as positional information.  





    
![svg](Blog_files/Blog_15_0.svg)
    



In code, the symbol `tokens` represents the simplest transform. It returns the tokens passed to the model. The default input is the sequence "hello". 

```
tokens
```



    
![svg](Blog_files/Blog_17_0.svg)
    



If we want to change the input to the transform, we use the input method to pass in an alternative. 

```
tokens.input([5, 2, 4, 5, 2, 2])
```



    
![svg](Blog_files/Blog_19_0.svg)
    



As with Transformers, we cannot access the positions of these sequences directly. However, to mimic position embeddings, we have access to a sequence of indices.

```
indices
```



    
![svg](Blog_files/Blog_21_0.svg)
    



```
sop = indices
sop.input("goodbye")
```



    
![svg](Blog_files/Blog_22_0.svg)
    



### Feed Forward Network

After the input layer, we reach the feed-forward network. In a Transformer, this stage can apply mathematical operations to each element of the sequence independently. 





    
![svg](Blog_files/Blog_24_0.svg)
    



In code, we represent this stage by computation on transforms. Mathematical operations are overloaded to represent independent computation on each element of the sequence .

```
tokens == "l"
```



    
![svg](Blog_files/Blog_26_0.svg)
    



The result is a new transform. Once constructed it can be applied to new input.

```
model = tokens * 2  - 1
model.input([1, 2, 3, 5, 2])
```



    
![svg](Blog_files/Blog_28_0.svg)
    



Operations can combine multiple transforms. For example, functions of `tokens` and `indices`. The analogy here is that the Transformer activations can keep track of multiple pieces of information simultaneously.

```
model = tokens - 5 + indices
model.input([1, 2, 3, 5, 2])
```



    
![svg](Blog_files/Blog_30_0.svg)
    



```
(tokens == "l") | (indices == 1)
```



    
![svg](Blog_files/Blog_31_0.svg)
    



We provide a few helper functions to make it easier to write transforms. For example, `where` provides an "if" statement like construct

```
where((tokens == "h") | (tokens == "l"), tokens, "q")
```



    
![svg](Blog_files/Blog_33_0.svg)
    



And `map` lets us define our own operators, for instance a string to int transform. (Users should be careful to only use operations here that could be computed with a simple neural network).

```
atoi = tokens.map(lambda x: ord(x) - ord('0'))
atoi.input("31234")
```



    
![svg](Blog_files/Blog_35_0.svg)
    



When chaining these transforms, it is often easier to work with functions. For example the following applies where and then <code>atoi</code> and then adds 2.

```
def atoi(seq=tokens):
    return seq.map(lambda x: ord(x) - ord('0')) 

op = (atoi(where(tokens == "-", "0", tokens)) + 2)
op.input("02-13")
```



    
![svg](Blog_files/Blog_37_0.svg)
    



From here on, unless we use a different input sequence, we will assume that the input is ‘hello’ and omit the input display in the illustrations.


###  Attention Selectors

Things get more interesting when we start to apply attention. This allows routing of information between the different elements of the sequence. 





    
![svg](Blog_files/Blog_39_0.svg)
    



We begin by defining notation for the keys and queries of the model. Keys and queries are effectively transforms that we will broadcast and compare to each other to create *selectors*, our parallel to attention patterns. We create them directly from transforms. For example, if we want to define a key, we call `key` on a transform.

```
key(tokens)
```



    
![svg](Blog_files/Blog_41_0.svg)
    



Similarly for `query`. (Queries are presented as columns to reflect their relation to the selectors we will create from them.)

```
query(tokens)
```



    
![svg](Blog_files/Blog_43_0.svg)
    



Scalars can be used as keys or queries. They broadcast out to the length of the underlying sequence.

```
query(1)
```



    
![svg](Blog_files/Blog_45_0.svg)
    



By applying a comparison operation between a key and a query we create a *selector*, our parallel to an attention matrix - though this one is unweighted. 

A selector is a binary matrix indicating which input position (column) each output position (row) will attend to in an eventual attention computation. In the comparison creating it, the key values describe the input (column) positions, and the query values describe the output (row) positions.

```
eq = (key(tokens) == query(tokens))
eq
```



    
![svg](Blog_files/Blog_47_0.svg)
    



Some examples: 

* A selector that matches each output position to the previous input position.

```
offset = (key(indices) == query(indices - 1))
offset
```



    
![svg](Blog_files/Blog_49_0.svg)
    



* A selector that matches each output position to all earlier input positions.

```
before = key(indices) < query(indices)
before
```



    
![svg](Blog_files/Blog_51_0.svg)
    



* A selector that matches each output position to all later input positions.

```
after = key(indices) > query(indices)
after
```



    
![svg](Blog_files/Blog_53_0.svg)
    



Selectors can be merged using boolean operations. For example, this selector focuses each output position on 1) earlier positions that 2) contain the same original input token as its own. We show this by including both pairs of keys and queries in the matrix.

```
before & eq
```



    
![svg](Blog_files/Blog_55_0.svg)
    



## Using Attention 

Given an attention selector we can provide a value sequence to aggregate. We represent aggregation by **summing** up over the values that have a true value for their selector. 

(Note: in the original paper, they use a **mean** aggregation and show a clever construction where mean aggregation is able to represent a sum calculation. RASPy uses sum by default for simplicity and to avoid fractions. In practicce this means that RASPy may underestimate the number of layers needed to convert to a mean based model by a factor of 2.)

Attention aggregation gives us the ability to compute functions like histograms. 

```
(key(tokens) == query(tokens)).value(1)
```



    
![svg](Blog_files/Blog_59_0.svg)
    



Visually we follow the architecture diagram. Queries are to the left, Keys at the top, Values at the bottom, and the Output is to the right.





    
![svg](Blog_files/Blog_61_0.svg)
    



Some attention operations may not even use the input tokens. For instance to compute the `length` of a sequence, we create a “select all” attention selector and then add 1 from each position.

```
length = (key(1) == query(1)).value(1)
length = length.name("length")
length
```



    
![svg](Blog_files/Blog_63_0.svg)
    



Here's a more complex example, shown step-by-step. (This is the kind of thing they ask in interviews!)

Say we want to compute the sum of neighboring values in a sequence, along a sliding window. First we apply the forward cutoff, attending only to positions that are not too far in the past.

```
WINDOW=3
s1 = (key(indices) >= query(indices - WINDOW + 1))  
s1
```



    
![svg](Blog_files/Blog_65_0.svg)
    



Then the backward cutoff, attending only to positions up to and including our own.

```
s2 = (key(indices) <= query(indices))
s2
```



    
![svg](Blog_files/Blog_67_0.svg)
    



Intersect.

```
sel = s1 & s2
sel
```



    
![svg](Blog_files/Blog_69_0.svg)
    



And finally aggregate.

```
sum2 = sel.value(tokens) 
sum2.input([1,3,2,2,2])
```



    
![svg](Blog_files/Blog_71_0.svg)
    



Here is a simple example that produces a 2-layer transform. The first corresponds to computing length and the second the cumulative sum. The cumulative sum has to go into a second layer because it is applied to a transform which uses length, and so it can only be computed after the computation of length is complete.

```
def cumsum(seq=tokens):
    x = (before | (key(indices) == query(indices))).value(seq)
    return x.name("cumsum")
cumsum().input([3, 1, -2, 3, 1])
```



    
![svg](Blog_files/Blog_73_0.svg)
    



## Layers 

The language supports building up more complex transforms. It keeps track of the *layers* by tracking the operations computed so far. 





    
![svg](Blog_files/Blog_76_0.svg)
    



Here is a simple example that produces a 2-layer transform. The first corresponds to computing length and the second the cumulative sum.

```
x = cumsum(length - indices)
x.input([3, 2, 3, 5])
```



    
![svg](Blog_files/Blog_78_0.svg)
    



## Coding with Transformers

Given this library of functions, we can write operations to accomplish surprisingly complex tasks. 

**Can we produce a Transformer that does basic addition of two arbitrary length numbers?**

i.e. given a string "19492+23919" can we produce the correct output? 

We will go through these steps, and their solutions, here. If you would rather do them on your own, we provide [a version](https://colab.research.google.com/github/srush/raspy/blob/main/Blog2.ipynb") where you can try them yourself!

Before we dive in to the main task, we will do some challenges of increasing difficulty to help us build some intuitions.


### Challenge 1: Select a given index

Produce a sequence where all the elements have the value at index i.

```
def index(i, seq=tokens):
    x = (key(indices) == query(i)).value(seq)
    return x.name("index")
index(1)
```



    
![svg](Blog_files/Blog_83_0.svg)
    



### Challenge 2: Shift

Shift all of the tokens in a sequence to the right by i positions. (Here we introduce an optional parameter in the aggregation: the default value to be used when no input positions are selected. If not defined, this value is 0.)

```
def shift(i=1, default="_", seq=tokens):
    x = (key(indices) == query(indices-i)).value(seq, default)
    return x.name("shift")
shift(2)
```



    
![svg](Blog_files/Blog_85_0.svg)
    



### Challenge 3: Minimum 

Compute the minimum values of the sequence. (This one starts to get harder. Our version uses 2 layers of attention.)

```
def minimum(seq=tokens):
    sel1 = before & (key(seq) == query(seq))
    sel2 = key(seq) < query(seq)
    less = (sel1 | sel2).value(1)
    x = (key(less) == query(0)).value(seq)
    return x.name("min")
minimum()([5,3,2,5,2])
```



    
![svg](Blog_files/Blog_87_0.svg)
    



The idea behind our solution is an implicit full ordering of the input positions: we (implicitly) order the positions according to input token value, with input position as tie breaker. Our first act is to have each position attend to all positions before it in the ordering: `sel1` focuses on earlier input positions with the same input token value, and `sel2` focuses on input positions with lower input token value. We then aggregate a 1 from all positions to get where each position is located in this ordering (i.e., how many other positions precede it). The minimum value is the input value at the first position according to this ordering (i.e., the one which had no other positions precede it).

### Challenge 4: First Index

Compute the first index that has token q, assuming the sequence always has length shorter than 100. (2 layers)

```
def first(q, seq=tokens):
    return minimum(where(seq == q, indices, 99))
first("l")
```



    
![svg](Blog_files/Blog_90_0.svg)
    



### Challenge 5: Right Align

Right align a padded sequence e.g. ralign().inputs('xyz___') = '---xyz'" (2 layers)

```
def ralign(default="-", sop=tokens):
    c = (key(sop) == query("_")).value(1)
    x = (key(indices + c) == query(indices)).value(sop, default)
    return x.name("ralign")
ralign()("xyz__")
```



    
![svg](Blog_files/Blog_92_0.svg)
    



### Challenge 6: Split

Split a sequence into two parts at value v and then right align. You can assume there is exactly one appearance of v in the sequence. (3 layers to get and align the first part of the sequence, but only 1 for the second.)

```
def split(v, get_first_part, sop=tokens, default="0"):
    split_point = (key(sop) == query(v)).value(indices)
    if get_first_part:
        x = ralign(default, 
                   where(indices < split_point, 
                         sop, "_"))
        return x
    else:
        x = where(indices > split_point, sop, default)
        return x
split("+", False)("xyz+zyr")
```



    
![svg](Blog_files/Blog_94_0.svg)
    



```
split("+", 0)("xyz+zyr")
```



    
![svg](Blog_files/Blog_95_0.svg)
    



### Challenge 6: Slide

Replace special tokens "<" with the closest non "<" value to their right. (2 layers)

```
def slide(match, seq=tokens):
    x = cumsum(match) 
    y = ((key(x) == query(x + 1)) & (key(match) == query(True))).value(seq)
    seq =  where(match, seq, y)
    return seq.name("slide")
slide(tokens != "<").input("xxxh<<<l")
```



    
![svg](Blog_files/Blog_97_0.svg)
    



### Challenge 7: Add

For this one you want to perform addition of two numbers. Here are the steps. 

```python
add().input("683+345")
```

0. Split into parts (challenge 6). Convert to ints. Add.

> “683+345” => [0, 0, 0, 9, 12, 8]

1. Compute the carry terms. Three possibilities: definitely receives carry (“1”), definitely doesn't receive carry (“0”), maybe receives carry (“<”).Because we are only adding two numbers, the only  case in which a position might receive a carry is if the position after it sums to 9. In that case, it will receive a carry if and only if the position after *that* receives a carry.

> [0, 0, 0, 9, 12, 8] => “00<100”

2. Slide the carry coefficients. A position that might receive a carry will get one if and only if the next position receives a carry - and so on down the chain until the next definite carry/no carry.

> “00<100” => 001100"

3. Complete the addition.

Each of these is 1 line of code. The full system is 6 layers. (if you are careful you can do it in 5!).


```
def add(sop=tokens):
    # 0) Parse and add
    x = atoi(split("+", True, sop)) \
        + atoi(split("+", False, sop))
    # 1) Check for carries 
    gets_carry = shift(-1, "0", where(x > 9, "1", where(x == 9, "<", "0")))
    # 2) Slide carries to their columns - all in one parallel go!                                         
    gets_carry = atoi(slide(gets_carry != "<", gets_carry))
    # 3) Add in carries, and remove overflow from original addition.                                                                                  
    return (x + gets_carry) % 10
add()("683+345")
```



    
![svg](Blog_files/Blog_99_0.svg)
    



```
683 + 345
```



    1028



Pretty neat stuff. If you are interested more in this topic, be sure to check at the paper: 

[Thinking like Transformers](https://arxiv.org/pdf/2106.06981.pdf) and the [RASP language](https://github.com/tech-srl/RASP).


If you're generally interested in the connection of formal languages and neural networks (FLaNN), or know people who are, there is an online community for exactly that, which has been very friendly and reasonably active so far!

Check it out at https://flann.super.site/ . 


This blog post, including the library, visualization and blog post is available at https://github.com/srush/raspy .

