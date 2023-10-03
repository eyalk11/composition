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
C / list / map % (lambda x: x*2) @ range(1,5)
# Output: [2, 4, 6, 8]
C / list / zip & C / list @ range(5) ^ [4,8,9,10,11]
# Output: [(0, 4), (1, 8), (2, 9), (3, 10), (4, 11)]
# More examples...
```
## Installation
To install the project, simply run the following command:
```
pip install composition 
```
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
