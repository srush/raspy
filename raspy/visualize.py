from raspy.rasp import mean2
from chalk import *
from raspy import *
from colour import Color
orange = Color("#f4c095")
green = Color("#679289")
red = Color("#ee2e31")
black = Color("#071e22")
blue = Color("#1d7874")

def word(x, c=Color("black")):
    if isinstance(x, bool):
        x = int(x)
    if x == 99:
        x = "-"
    return text(x, 0.85).with_envelope(rectangle(1, 1)).fill_color(c).line_width(0)

def draw_sel(self):
    top =  empty() #hcat([word(x, Color("lightgrey")) for x in  self.bind])
    left = empty()
    for k, q in self.kqs:
        top = top / hcat([word(x, orange) for x in q.val])
        left = left | vcat([word(x, green) for x in k.val], 0)
        
    mat = self.val
    heat =  vcat([hcat([rectangle(1, 1).line_color(black).fill_color(Color("lightgrey") if x else Color("white"))
                        for x in row]) for row in mat]).frame(0.1)
    d = heat.translate(2, 0).center_xy().beside(top.center_xy(), -unit_y)
    d = d.beside(left.center_xy(), -unit_x) 
    return d

def draw(bself):
    
    self = bself.sel
    d = draw_sel(self)
    
    if any([isinstance(r, float) for r in bself.result]):
            result = [
                    mean2(
                        [bself.v.val[j] for j in range(len(bself.v.val)) if self.val[j][i]]
                    )
                    for i in range(len(self.val))
                ]
            bottom = hcat([word(x[0]) for x in result])
            bottom = bottom / hcat([word("-") for x in result])
            bottom = bottom / hcat([word(x[1]) for x in result])
    
    else:
        bottom = hcat([word(x, black) for x in bself.result])
    if not bself.width:
        right = vcat([word(x, red) for x in bself.v.val])
    else:
        right = vcat([])
        
    d = d.beside(right.center_xy(), unit_x)
    d = d.beside(bottom.center_xy(), unit_y)

    if bself._name:
        d = d.beside(word("(" + bself._name + ")", Color("lightgrey")).center_xy(), -unit_y)
    return d

def draw_all(seq):
    initial = hcat([word(v) for v in seq.bind])
    final = hcat([word(v) for v in seq.val])
    sels = sorted(seq.sels, key=lambda x: x.layer)
    d  = {}
    for sel in sels:
        d.setdefault(sel.layer, [])
        d[sel.layer].append(sel)
    
    r = rectangle(1, 1).frame(0.5).line_color(Color("lightgrey"))
    r = r.beside(word("Q"), -unit_y).beside(word("K"), -unit_x).beside(word("V"), unit_x).beside(word("="), unit_y)
    dia =  vcat(
        [word("Input", blue).with_envelope(rectangle(3, 1)) | initial] + 
        [word("Layer " + str(k), blue).with_envelope(rectangle(3, 1)) | hstrut(1.0) | hcat([draw(sel) for sel in v], 1)
        for k, v in d.items()] +
        [word("Final", blue).with_envelope(rectangle(3, 1)) | final], 1).center_xy()
    # if len(d) > 0:
    #     dia = dia | hstrut(1) | r
    return dia
