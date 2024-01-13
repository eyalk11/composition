# Composition

## Description
The project is a Python library that allows for composition with currying. It provides a set of operators and functions that enable functional programming techniques such as partial function application and composition. With this library, you can easily compose functions, apply them to data, and manipulate collections in a functional style.

## Features
- Allows for composition with currying
- Supports partial function application
- Supports function assignment
- Provides various operators for composition and application

## Usage

```python
from composition import C


C / list / map % (lambda x: x*2) @ range(1,5)
# Output: [2, 4, 6, 8]
# Instead of list(map(lambda x:x*2, range(1,5))) 
```


So, `/` is composition , `%` is partial and `@` is applying.

`(f / g)(x)` is `f(g(x))`




### Variables and piping
You can work with [sspipe](https://github.com/sspipe/sspipe) to do nice things.
```
C / X.items() * {'a': 'b', 'c': 'a'}
```
Also call functions. Variable X is special variable from pipe.
There are the following special variables: 

- X,Y,Z are taken from tuple of current elements if there is one.
- Orig means that it is an original argument of the pipe 

So arguments can be passed to functions along the chain.

An example:

```python
def load_user(user_id: int, db: sqlite3.Connection) -> User:
    ...


def find_computer( db : sqlite3.Connection,user: User) -> str:
    ...

C / find_computer % A(db=Orig,user=X) / load_user
```

You can use class `A` to specify arguments. This way it would know the context.

Lets change it a bit.

```python
def load_user(user_id: int, db ):
    return User(user_id)
x= C.find_computer(db=Orig,computer=X.computer) / load_user @ ('a','b')
```
Would also work.

## More on partial (and function)

Partial can be done with class a or with dictionary

```python
def do(a,b):
    pass
C / do % {'a':3} @ 6
```
Or

```python
C / find_computer % A(db=Orig,user=X) / load_user
```
class `A` supports pipes.


Partial supports function assigment, which means you can do:

```python
 def f(a,b,c):
    return (a,b,c)
 f % {'b': "->b*a*2"} @ (1,3)
 or f (lambda b: b*2}
``` 
We supports syntax `->b*a*2` to suggest a function.
But that is not safe.

#### Notice

The implementation can be unsafe because it converts string to functions. 
So on undistilled input, please use `CS` instead of `C`. which doesn't allow it.
(otherwise an attacker might be able to inject a function in a string starting with ->).


## Working with collections.

You can start from a collection and do << to apply function on each element (i.e. map).
```python
x= (C/[1,2,3] << '-> x*3') << (lambda x:x*3) | exp
```
When you are done , you can do `| exp` or `| explist` to convert to expression(generator or list, depends on you) or list

## Operator precedence

```python
C / list / zip & C / list @ range(5) ^ [4,8,9,10,11]
# Output: [(0, 4), (1, 8), (2, 9), (3, 10), (4, 11)]
```

& is partial but in lower precedence .
^ is applying with lower precedence

Another example:
```python
C / set / reduce % (lambda x,y:x+y) @ (C /  self._hist_by_date.values() << (lambda s: list(s.keys())) )
```




## Installation
To install the project, simply run the following command:
```
pip install composition 
```


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
