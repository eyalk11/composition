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


`f% (lambda x:x*2)` sets the first variable to x*2. But it can also get a dictionary.




```python
C / list / zip & C / list @ range(5) ^ [4,8,9,10,11]
# Output: [(0, 4), (1, 8), (2, 9), (3, 10), (4, 11)]
```

& is partial but in lower precedence .
^ is applying with lower precedence


```python
C / set / reduce % (lambda x,y:x+y) @ (C /  self._hist_by_date.values() << (lambda s: list(s.keys())) )
#The same as set(reduce(lambda x,y:x+y, [list(s.keys()) for s in self._hist_by_date.values()]) )
```



Partial supports function assigment, which means you can do:

```python
 def f(a,b,c):
    return (a,b,c)
 f % {'b': "->b*a*2"} @ (1,3)
 or f (lambda b: b*2}
``` 

It also supports currying. 

The term is used freely here. It means that the original arguments can be passed to function along the chain, if it serves the purpose. 

```python
def load_user(user_id: int, db: sqlite3.Connection) -> User:
    ...


def find_computer( db : sqlite3.Connection,user: User) -> str:
    ...
```

consider the above chain.  db is  common parameters to both. 

Now you can call specify arugment to take from ret of previous function (X,Y,Z by the order) , or just use Orig to take the same parameter from original call. 

```python
C / find_computer % A(db=Orig,user=X) / load_user
```

## Installation
To install the project, simply run the following command:
```
pip install composition 
```

## Notice

The implementation can be unsafe because it converts string to functions. 
So on undistilled input, please use `CS` instead of `C`. 
(otherwise an attacker might be able to inject a function in a string starting with ->).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
