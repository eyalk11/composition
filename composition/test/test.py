
from composition import C, Orig, X, A, exp, CS


def f(a,b,c):
    return (b)
def g(b,c):
    print(c)
    return (b)
def h(b,c):
    print(c)
    return (b)

def test_dics():
    def r():
        return {'a':'b','c':'a'}

    x= CS // sorted % { 'key': lambda x:x[1]} / list  * (r().items())
    assert x == [('c', 'a'), ('a', 'b')]

    x = C / sorted / A(key=lambda x: x[1]) / list / X.items()  * (r())
    assert x == [('c', 'a'), ('a', 'b')]
    z = C / X.items() * {'a': 'b', 'c': 'a'}
    assert 'dict_items' in str(type(z))


# def test_curry():
#     x=  (C // h // g // f @ (3, 7,8))
#     assert x==7


def test_col():
    x= (C/[1,2,3] << '-> x*3') << (lambda x:x*3) | exp
    assert list(x)==[9,18,27]
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
    class User:
        def __init__(self, computer):
            self.computer = computer
    def find_computer(db , computer : str):
        print(db,computer)
    def load_user(user_id: int, db ):
        return User(user_id)
    #x=C / find_computer % A(db=Orig,computer=X) / load_user @ ('a','b')
    x= C.find_computer(db=Orig,computer=X.computer) / load_user @ ('a','b')
    print(x)

