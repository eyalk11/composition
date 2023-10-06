from composition import C


def f(a,b,c):
    return (b)
def g(b,c):
    print(c)
    return (b)
def h(b,c):
    print(c)
    return (b)

def test_curry2():
    def r():
        return {'a':'b','c':'a'}

    x= C // sorted // dict @ { 'key': (lambda x:x[1]), 'seq': r()  }
    assert x == {'c': 'a', 'a': 'b'}


def test_curry():
    x=  (C // h // g // f @ (3, 7,8))
    assert x==7


def test_col():
    x= (C/[1,2,3] << '-> x*2') % 0
    assert list(x)==[2,4,6]
def test_basic():
    f= C / list /map % (lambda x: x*2) @  range(1,5)
    assert f==[2,4,6,8]
    assert list(f)==[2,4,6,8]

def test_tmp():
    x= C/ (1,2,3) / list %2
    assert x==[1,2]
def test_colb():
    x= C/(1,8,2) @ range
    assert x==range(1,8,2)
def test_curry_part():
    d= C / f % {'b': (lambda b:b*2)} @ (1,2,3)
    assert d==(4)
    d = C / f % {'b': '->(a+b)*2'} @ (1, 2, 3)
    assert d == (6)
def test_adv():
    x=C / list / zip & C / list @ range(5) ^ [4, 8, 9, 10, 11]
    assert x==[(0, 4), (1, 8), (2, 9), (3, 10), (4, 11)]
def test_cur_adv():
    def find_computer(db , computer : str):
        print(db,computer)
    def load_user(user_id: int, db ):
        return user_id
    x=C // (find_computer % {'computer': 'v' }) // load_user @ ('a','b')

def test_pipe():
    from sspipe import p, px
    import numpy as np
    import matplotlib.pyplot as plt

    y={"a": "b","b":"c"} | px.get(px['a'])
    y=y
