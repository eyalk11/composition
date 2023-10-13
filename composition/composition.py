from __future__ import annotations
import itertools
import typing
import warnings

from copy import copy

from typing import ParamSpec

from typing import Callable, Generic, TypeVar
from enum import Enum, Flag

import inspect
from functools import partial
from multimethod import overload as singledispatchmethod, overload as singledispatch

from inspect import _ParameterKind, signature

import itertools
import typing
from copy import copy
from inspect import Signature

from typing import ParamSpec

from typing import Callable, Generic, TypeVar
from enum import Enum, Flag

import inspect
from functools import partial
from multimethod import overload as singledispatch, overload as singledispatchmethod

from inspect import _ParameterKind, signature

from sspipe.pipe import Pipe
from sspipe.pipe import _resolve

dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
dictnfilt = lambda x, y: dict([(i, x[i]) for i in x if not (i in set(y))])

T = TypeVar('T')
P = ParamSpec('P')
Q = ParamSpec('Q')
U = TypeVar('U')


class UnsafeType(Flag):
    NotSafe = 0
    Safe = 1
    WithFuncs = 2
    Currying = 4


class MyPipe(Pipe):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name


Y = MyPipe(func=lambda x: x, name="Y")
X = MyPipe(func=lambda x: x, name="X")
Z = MyPipe(func=lambda x: x, name="Z")


class A():
    def __init__(self, *args, **kw):
        self.constargs = args
        self.constkw = kw

    def resolve(self, sig: Signature, origargs, args):
        bconst = sig.bind_partial(*self.constargs, **self.constkw)

        for k, v in bconst.arguments:
            if isinstance(v, MyPipe):
                if v.name == 'X':
                    arg = 0
                elif v.name == 'Y':
                    arg = 1
                elif v.name == 'Z':
                    arg = 2
                else:
                    arg = None
                if arg > len(args.args):
                    raise "not enough args"

                v = _resolve(v, args.args[arg] if arg is not None else args.args)


            elif type(v) is Orig:
                if origargs.arguments is not None:
                    if k in origargs.arguments:
                        v = origargs.arguments[k]
                else:
                    warnings.warn(f"origargs is none. Cant resolve {k}")
                yield k, v


class Exp:
    pass


class ExpList(Exp):
    pass


class Orig:
    pass


class CallWithArgs():

    def __init__(self, other, func):
        self._arguments = None
        try:
            s = signature(func)
        except ValueError:
            if (type(other) is tuple):
                self._args = other
                self._kwargs = {}
            elif type(other) == dict:
                self._kwargs = other
                self._args =tuple()
            else:
                self._args = (other,)
                self._kwargs={}
        else:
            if (type(other) in [tuple, list]):
                b = s.bind_partial(*other)
            elif (type(other) == dict):
                b = s.bind_partial(**other)
            else:
                b = s.bind_partial(other)

            self._arguments = b.arguments
            self._kwargs = b.kwargs
            self._args = b.args

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def arguments(self):
        return self._arguments


class CInst(Generic[P, T]):
    @singledispatch
    def __init__(self, func: Callable, prev: CInst = None, unsafe: UnsafeType = UnsafeType.NotSafe,
                 a: typing.Optional[A] = None):
        self.func = func
        self._unsafe = unsafe
        self.prev = prev
        self.origargs = None
        self.col = None
        self._a = a

    @__init__.register
    def __init(self, other: typing.Union[typing.Collection, typing.Generator], unsafe: UnsafeType = UnsafeType.NotSafe):
        self.col = other
        self._unsafe = unsafe

    @singledispatchmethod
    def __floordiv__(self, other: Callable) -> CInst[Q, T]:
        return CInst(other, self, self._unsafe | UnsafeType.Currying)

    @__floordiv__.register
    def __floordiv__(self, other: str) -> CInst[Q, T]:
        if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
            return self.__floordiv__(CInst.conv_str_to_func(other), self._unsafe)
        else:
            raise "cant do it when safe"

    def __iter__(self):
        if self.col is None:
            raise ValueError('Can only use iter  on collection')

        return iter(self.col)

    def __len__(self):
        if self.col is None:
            raise ValueError('Can only use len  on collection')
        return len(self.col)

    def __getitem__(self, item):
        if self.col is None:
            raise ValueError('Can only use getitem  on collection')
        return self.col[item]

    def __eq__(self, other):
        if type(other) is CInst:
            return self.func == other.func and self.col == other.col

        if self.col is not None:

            if inspect.isgenerator(self.col):
                xx, yy = itertools.tee(self.col)
                try:
                    for t in zip(yy, other):
                        if t[0] != t[1]:
                            return False
                    return True
                finally:
                    self.col = xx
            return self.col == other
        return self.func == other

    def apply_with_a(self, origargs: CallWithArgs, args: typing.Any) -> T:
        if self.prev.func is None:
            raise ValueError("cant apply with a when prev is none")
        if self._a is None:
            raise ValueError("cant apply with a when a is none")
        bargs = CallWithArgs(args, self.prev.func)
        sig = signature(self.prev.func)

        addargs = {k: v for (k, v) in self._a.resolve(sig, origargs,bargs)}
        args_for_partial, kwargs_for_partial = CInst.update_args_from_additional(addargs, tuple(), {}, sig)
        nf = partial(self.prev.func, *args_for_partial, **kwargs_for_partial)
        self.prev.func = nf
        return self.prev.apply_int(origargs, args)

    @staticmethod
    def update_args_from_additional(add_args, bargs, kwargs, sig):
        for parameter in add_args.keys():

            s = sig.parameters[parameter]
            if s.kind == _ParameterKind.POSITIONAL_ONLY:
                bargs = tuple(list(bargs) + [add_args[parameter]])
            else:
                kwargs.update({parameter: add_args[parameter]})
        return bargs, kwargs

    def apply_int(self, origargs: CallWithArgs, args: typing.Any) -> T:

        if self._a is not None:
            return self.apply_with_a(origargs, args)

        if self.func is None:
            return args

        args = CallWithArgs(args, self.func)

        res = self.func(*args.args, **args.kwargs)
        if self.prev is None:
            return res

        return self.prev.apply_int(origargs, res)

        # if self._unsafe & UnsafeType.Currying != UnsafeType.Currying:
        # return basic(bargs, bkwargs)
        # else:
        # try:
        # sig = signature(self.func)
        # b = sig.bind(*bargs, **bkwargs)
        # except TypeError:

        # add_args= dictfilt(dictnfilt(origargs,b.arguments),sig.parameters.keys())
        # bargs ,bkwargs= CINST.update_args_from_additional(add_args, bargs, bkwargs, sig)

        # return basic(bargs, bkwargs)
        # return basic(bargs, bkwargs)

    def __or__(self, other):
        if self.col is None:
            raise ValueError('Can only use |  on collection')
        if type(other) == Exp:
            return self.col
        elif type(other) == ExpList:
            return list(self.col)

    @singledispatch
    def __truediv__(self, other: Callable) -> CInst:
        if self.col is not None:
            return CInst(other(self.col), unsafe=self._unsafe)
        return CInst(other, self, self._unsafe & (~UnsafeType.Currying), None)

    @__truediv__.register
    def __truediv__(self, other: typing.Collection) -> CInst[Q, T]:
        return CInst(other, self._unsafe)

    @__truediv__.register
    def __truediv__(self, other: str) -> CInst[Q, T]:
        if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
            return self.__truediv__(CInst.conv_str_to_func(other))
        else:
            raise "cant do it when safe"

    def __and__(self, other):
        return self.__mod__(other)

    def __xor__(self, other):
        return self.__matmul__(other)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

    @singledispatchmethod
    def __matmul__(self, other) -> T:

        return self.apply(other)

    def apply(self, other):
        return self.apply_int(CallWithArgs(other, self.func), other)

    @__truediv__.register
    def __truediv__(self, other: A):
        return CInst(a=other, prev=self, unsafe=self._unsafe)

    @__matmul__.register
    def __matmul__(self, other: typing.Callable) -> T:
        if self.col is None:
            return self.apply(other)
        if type(self.col) is dict:
            return other(**self.col)
        else:
            return other(*self.col)

    @staticmethod
    def conv_str_to_func(st):
        if st.startswith('->'):
            return eval('lambda x:' + st[2:])

    @__matmul__.register
    def __matmul__(self, other: str) -> T:
        if self.col is None:
            return self.apply(other)
        if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
            return self.__matmul__(CInst.conv_str_to_func(other))
        else:
            raise "cant do it when safe"

    @singledispatchmethod
    def __lshift__(self, other: Callable):
        if self.col is None:
            raise ValueError('Can only use << on collection')

        def gen():
            for k in self.col:
                yield other(k)

        return CInst(gen(), self._unsafe)

    @__lshift__.register
    def __lshift__(self, other: str):
        return self.__lshift__(CInst.conv_str_to_func(other))

    def __mod__(self, other):
        def handle_currying():
            origsig = signature(self.func)
            f = partial(self.func, **other)
            sig = signature(f)
            newparams = {}
            mapped_to_func = {}
            nother = other.copy()
            s = set()
            lambda_dic = dict()
            if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
                for k, v in other.items():
                    if type(v) == str and v.startswith('->'):
                        lambda_dic[k] = f'{v.replace("->", "")}'

            # We start from regular
            for k, v in sig.parameters.items():

                if inspect.isfunction(v.default):
                    s.add(k)
                    p = signature(v.default).parameters
                    newparams.update(p.items())
                    mapped_to_func[k] = (v.default, set([k for k in p]))
                    nother.pop(k)
                elif k in lambda_dic:
                    nother.pop(k)
                    newparams[k] = v
                else:
                    newparams[k] = v
            origparams = {k: v for k, v in sig.parameters.items() if k not in s}
            newparams.update(origparams)

            # partial makes some args keyword only
            for k, v in origparams.items():
                if v.kind == _ParameterKind.KEYWORD_ONLY:
                    if origsig.parameters[k].kind == _ParameterKind.POSITIONAL_OR_KEYWORD:
                        newparams[k] = inspect.Parameter(k, _ParameterKind.POSITIONAL_OR_KEYWORD,
                                                         default=origsig.parameters[k].default,
                                                         annotation=origsig.parameters[k].annotation)

            nf = partial(self.func, **nother)  # determined values

            newsig = sig.replace(parameters=newparams.values())

            def newfunc(*aargs, **kwargs):
                b = newsig.bind(*aargs, **kwargs)
                b.apply_defaults()
                ntobind = {}

                for k, v in mapped_to_func.items():
                    func, args = v
                    dic = (dictfilt(b.arguments, args))
                    ntobind[k] = func(**dic)  # we removed from nf the
                for k, v in lambda_dic.items():
                    s.add(k)
                    print(v)
                    ntobind[k] = eval(v, b.arguments)

                for k, v in b.arguments.items():
                    if k not in ntobind:
                        ntobind[k] = v

                nd = dictfilt(ntobind, signature(nf).parameters.keys())
                return nf(**nd)

            return newfunc

        if self.col is not None:
            if other == 0:
                return self.col
            elif other > 0:
                return list(self.col)[:other]
            elif type(other) == slice:
                return list(self.col)[other]

        if (type(other) == tuple):
            fn = partial(self.func, *other)
        elif (type(other) == dict):

            fn = handle_currying() if self._unsafe & UnsafeType.WithFuncs == UnsafeType.WithFuncs else partial(
                self.func, **other)


        else:
            fn = partial(self.func, other)
        return CInst(fn, self.prev, unsafe=self._unsafe)


class CSimpInst(CInst):
    def __init__(self, unsafe=UnsafeType.NotSafe | UnsafeType.WithFuncs):
        self.func = None
        self._unsafe = unsafe
        self.prev = None
        self.col = None
        self._a=None

    def __call__(self):
        raise NotImplementedError()


C = CSimpInst()
CS = CSimpInst(unsafe=UnsafeType.Safe)
exp = Exp()
explist = ExpList()

# U = TypeVar('U')
