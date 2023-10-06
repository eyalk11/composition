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




```
C / list / zip & C / list @ range(5) ^ [4,8,9,10,11]
# Output: [(0, 4), (1, 8), (2, 9), (3, 10), (4, 11)]
```

& is partial but in lower precedence .
^ is applying with lower precedence


``` 
C / set / reduce % (lambda x,y:x+y) @ (C /  self._hist_by_date.values() << (lambda s: list(s.keys())) )
#The same as set(reduce(lambda x,y:x+y, [list(s.keys()) for s in self._hist_by_date.values()]) )
```




``` 
Partial supports function assigment which means you can do:
 def f(a,b,c):
    return (a,b,c)
 f % {'b': "->b*a*2"} @ (1,3)
 or f (lambda b: b*2}
``` 

`//` is applying with currying. 

The term is used freely here. It means that the original arguments can be passed to function along the chain, if it serves the purpose. 

``` 
def load_user(user_id: int, db: sqlite3.Connection) -> User:
    ...


def find_computer(computer : str, db : sqlite3.Connection):
    ...

C // find_computer // '-> x.Computer' //  load_user @ (user_id,db)
``` 

In this case, db was common parameters to both. And since only computer wasn't enough for the function to apply, it added db to the default. 

Notice however that it depends on the order of the arguments.
If the order would have been reversed: 
```
def find_computer(db : sqlite3.Connection, computer : str):

```

We would need an helper function. 

``` 
C // find_computer // lambda x,db : (db,x)  ... 
``` 

So it is still not ideal, and I wonder if it can be.. 

Another approach would be: 
```
C / find_computer % { 'computer' : (lambda user: user.Database ) } / load_user ..  

```

C / find_computer % A(db=ORIG,computer=X) / load_user / X.load_db << [1,2,3,4]

```


Also notice that by default it uses x as the argument for lambda.

## Installation
To install the project, simply run the following command:
```
pip install composition 
```
It has only one dependency (multimethod). 

## Notice

The implementation can be unsafe because it converts string to functions. 
So on undistilled input, please use `CS` instead of `C`. 
(otherwise an attacker might be able to inject a function in a string starting with ->).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
