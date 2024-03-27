from __future__ import annotations

import itertools
import threading
import typing
import warnings
from contextvars import ContextVar,Context


from typing import ParamSpec
from typing import Callable, Generic, TypeVar
from enum import Flag

import inspect
from functools import partial
from multimethod import overload as singledispatch, overload as singledispatchmethod


from inspect import _ParameterKind, signature

import itertools
import typing
from inspect import Signature
from typing import ParamSpec

from typing import Callable, Generic, TypeVar
from enum import Flag

import inspect
from functools import partial
from multimethod import overload as singledispatch, overload as singledispatchmethod

from inspect import _ParameterKind, signature

from sspipe.pipe import Pipe,_resolve


dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
dictnfilt = lambda x, y: dict([(i, x[i]) for i in x if i not in set(y)])

T = TypeVar('T')
P = ParamSpec('P')
Q = ParamSpec('Q')
U = TypeVar('U')

class UnsafeType(Flag):
    NotSafe = 0
    Safe = 1
    Currying = 2



Y = Pipe(func=lambda x: x, name="Y")
X = Pipe(func=lambda x: x, name="X")
Z = Pipe(func=lambda x: x, name="Z")
Orig= Pipe(func=lambda x: x, name="Orig") 

XVar=ContextVar('X')
YVar=ContextVar('Y')
ZVar=ContextVar('Z')
OrigVar=ContextVar('Orig')
TLocal=ContextVar('TLocal')


class A:
    '''
    Can be used in the pipe to pass arguments to the function
    You can `C / f/ A(*args,**kw)` would be similar to partial. But notice that you can use A(arg=X) to specify that the argument is from the pipe.
    Or you can use A(arg=Orig) to specify that it would be taken from an argument to the original function, before all pipes. 
    '''
    def __init__(self, *args, **kw):
        self.constargs = args
        self.constkw = kw
    @staticmethod 
    def get_arg_by_name(v):
        if v == 'X':
            arg = 0
        elif v == 'Y':
            arg = 1
        elif v == 'Z':
            arg = 2
        else:
            arg = None
        return arg

    def resolve(self, sig: Signature, origargs, args):
        def resolve_regarg(k,v):
            nonlocal used_set
            arg = A.get_arg_by_name(v._name)
            if arg is not None:
                if arg > len(args.args):
                    raise Exception("not enough args")
                used_set.add(arg)
            return  args.args[arg] if arg is not None else args.args

        def resolve_int():
            f=dict(zip([XVar,YVar,ZVar][:len(args.args)],args.args))
            for k,v in f.items():
                k.set(v)
            loc=threading.local()
            loc.usedvars = set()
            TLocal.set(loc)


            bconst = sig.bind_partial(*self.constargs, **self.constkw)
            dic={}
            for k, v in bconst.arguments.items():
                if isinstance(v, Pipe):
                    if hasattr(v,'_name') and v._name == "Orig":
                        if origargs.arguments is not None:
                            if k in origargs.arguments:
                                toresolve= origargs.arguments[k]
                                OrigVar.set(toresolve)
                            else:
                                raise ValueError(f"origargs doesnt contain it. Cant resolve {k}")
                        else:
                            raise ValueError(f"origargs arguments not found. Cant resolve {k}")
                    else:
                        toresolve= resolve_regarg(k,v)

                    v = _resolve(v, toresolve )
                dic[k]=v

            used_set.update(filter(lambda x: x is not None, map(A.get_arg_by_name,loc.usedvars)))
            args.removearg(used_set)

            for k in loc.usedvars:
                args.removekwarg(k.lower())
            return dic

        used_set = set()
        ctx=Context()
        return ctx.run(resolve_int) # so that only the vars will be available in the context






class Exp:
    '''
    converts  CInst to an expression when used after | 
    '''

    pass


class ExpList(Exp):
    '''
    converts  CInst to a class when used after |
    '''
    pass



class CallWithArgs():


    @singledispatchmethod
    def __init__(self, other, func : Callable):

        self._arguments = None
        if func is None:
            return
        try:
            s = signature(func)
        except (ValueError,TypeError):
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

    @__init__.register
    def __init__(self, a :  A,func: Callable):
        self.__init_b(func, a.constargs, a.constkw)

    @__init__.register
    def __init__(self, func : Callable, args : typing.Tuple, kwargs : typing.Dict):
        self.__init_b(func, args, kwargs)
    def __init_b(self, func : Callable, args : typing.Tuple, kwargs : typing.Dict):
        self._args=args
        self._kwargs=kwargs
        self.func=func
        self.init_internal(args, kwargs)

    def init_internal(self, args,  kwargs):
        try:
            s = signature(self.func)
        except ValueError:
            self._arguments = None
        except TypeError:
            self._arguments = None
        else:
            try:
                b = s.bind_partial(*args, **kwargs)
            except:
                warnings.warn("didnt work , will try to change sig")
                newparams = s.parameters.copy()
                for k, v in s.parameters.items():
                    if v.kind != _ParameterKind.POSITIONAL_OR_KEYWORD:
                        newparams[k] = inspect.Parameter(k, _ParameterKind.POSITIONAL_OR_KEYWORD, default=s.parameters[k].default, annotation=s.parameters[k].annotation)
                s=s.replace(parameters=newparams.values())
                b = s.bind_partial(*args, **kwargs)


            self._arguments = b.arguments
            self._kwargs = b.kwargs
            self._args = b.args

    def removearg(self,ids):
         self._args= tuple([k for i,k in enumerate(self._args) if i not in ids])





    def removekwarg(self,k):
        if k in self._kwargs:
            self._kwargs.pop(k)
        else:
            return
            if not self.arguments:
                raise ValueError("Cant do it")
            try:
                self._arguments.pop(k)
                self.init_internal(**self._arguments)
            except:
                raise ValueError("Cant do it2")

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


    @singledispatchmethod
    def chktype(self,x:typing.Union[typing.Collection, typing.Generator]):
        return True
    @chktype.register
    def chktype(self,x:typing.Callable):
        return False

    def __init__(self, other: typing.Union[typing.Collection, typing.Generator, Callable],prev : CInst = None,
               unsafe: UnsafeType = UnsafeType.NotSafe | UnsafeType.Currying,a: typing.Optional[A] = None, isshift=False):
        self.func = None
        self.col = None
        if other is not None:
            if not self.chktype(other):
                self.func = other
            else:
                self.col = other



        self._unsafe = unsafe
        self.prev = prev
        self.origargs = None
        self._a = a
        self.is_shift=isshift


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

    def apply_with_a(self, origargs: CallWithArgs, args: typing.Any,simple=False) -> T:
        if self.prev.func is None:
            raise ValueError("cant apply with a when prev is none")
        if self._a is None:
            raise ValueError("cant apply with a when a is none")
        if simple:
            bargs = CallWithArgs(self.prev.func, args=(args,), kwargs={})
        else:
            bargs = CallWithArgs(args, self.prev.func)

        #bargs = CallWithArgs(args, self.prev.func)
        sig = signature(self.prev.func)

        addargs =  self._a.resolve(sig, origargs,bargs)

        args_for_partial, kwargs_for_partial = CInst.update_args_from_additional(addargs, tuple(), {}, sig)
        nf = partial(self.prev.func, *args_for_partial, **kwargs_for_partial)
        self.prev.func = nf


        return self.prev.apply_int_raw(origargs, bargs.args, bargs.kwargs)

    @staticmethod
    def update_args_from_additional(add_args, bargs, kwargs, sig):
        for parameter in add_args.keys():

            s = sig.parameters[parameter]
            if s.kind == _ParameterKind.POSITIONAL_ONLY:
                bargs = tuple(list(bargs) + [add_args[parameter]])
            else:
                kwargs.update({parameter: add_args[parameter]})
        return bargs, kwargs



    def apply_int(self, origargs: CallWithArgs, args: typing.Any,simple=False) -> T:

        if self._a is not None:
            return self.apply_with_a(origargs, args,simple=simple)


        bargs = self.decode_args(args, simple)
        if bargs is None:
            return args

        res = self.func(*bargs.args, **bargs.kwargs)
        if self.prev is None:
            return res

        return self.prev.apply_int(origargs, res,simple=True)



    def decode_args(self, args, simple):
        if self.func is None:
            return None
        if type(args) is not CallWithArgs:
            if simple:
                bargs = CallWithArgs(self.func, args=(args,), kwargs={})
            else:
                bargs = CallWithArgs(args, self.func)
        else:
            bargs = args
        return bargs

    def apply_int_raw(self, origargs, args : typing.Tuple, kwargs: typing.Dict):
        if self._a is not None:
            warnings.warn("A in this situation?")


        if self.func is None:
            warnings.warn("Strange")
            return args

        res = self.func(*args, **kwargs)
        if self.prev is None:
            return res

        return self.prev.apply_int(origargs, res)






    def __or__(self, other):
        '''
        Turn collection to expression (genrator or list) or apply filter
        '''
        if self.col is None:
            raise ValueError('Can only use |  on collection')
        if type(other) == Exp:
            return self.col
        elif type(other) == ExpList:
            return list(self.col)
        elif type(other) == str:
            return CInst(filter(conv_str_to_func(other), self.col),self,self._unsafe) 
        else:
            return CInst(filter(other, self.col),self,self._unsafe)
        #elif isinstance(other,Pipe ): #we do pipe
#   inst=Pipe(lambda x: _resolve(self, x) | other)



    @singledispatch
    def __truediv__(self, other: Callable) -> CInst:
        '''
        Applies composition. 
        Can also be used to specify args with A / pipe. 
        Examples:  a. C / X.items() * {'a': 'b', 'c': 'a'}
                   b. C / list / zip & C / list @ range(5) ^ [4, 8, 9, 10, 11]
                   c. C / sorted / A(key=lambda x: x[1]) / list / X.items()  * (r())
        '''
        if self.col is not None:
            return CInst(other(self.col), unsafe=self._unsafe)
        return CInst(other, self, self._unsafe & (~UnsafeType.Currying), None)

    @singledispatchmethod
    def __floordiv__(self, other: Callable) -> CInst[Q, T]:
        return CInst(other, self, self._unsafe | UnsafeType.Currying)

    @__floordiv__.register
    def __floordiv__(self, other: str) -> CInst[Q, T]:
        if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
            return self.__floordiv__(CInst.conv_str_to_func(other), self._unsafe)
        else:
            raise Exception("cant do it when safe")

    @__truediv__.register
    def __truediv__(self, other: Pipe) -> CInst[Q, T]:
        return self/ (lambda x:x) / A(x=other)

    @__truediv__.register
    def __truediv__(self, other: Exp) -> CInst[Q, T]:
        return self | exp

    @__truediv__.register
    def __truediv__(self, other: typing.Collection) -> CInst[Q, T]:
        return CInst(other, self._unsafe)

    @__truediv__.register
    def __truediv__(self, other: str) -> CInst[Q, T]:
        if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
            return self.__truediv__(CInst.conv_str_to_func(other))
        else:
            raise "cant do it when safe"

    @__truediv__.register
    def __truediv__(self, other: A):
        return CInst(None, self, self._unsafe,other)

    def __and__(self, other):
        return self.__mod__(other)

    def __xor__(self, other):
        return self.__matmul__(other)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.apply_int_raw(CallWithArgs(self.func,args,kwargs), args, kwargs)

    @singledispatchmethod
    def __matmul__(self, other) -> T:

        return self.apply(other)

    def apply(self, other,simple=False):

        return self.apply_int( args if (args:= self.decode_args(other,False)) is not None else other , other,simple=simple)

    @staticmethod
    def conv_str_to_func(st):
        if st.startswith('->'):
            return eval('lambda x:' + st[2:])

    '''
    Simple apply. Take what on the other side as the firest argument to the function.
    C/ list * [1,2,3] 
    '''
    def __mul__(self, other):
        args = self.decode_args(other,True)
        if args is None:
            args=other
        return self.apply_int(args,other,True)

    '''
    Regular apply. If it is a dict, uses **kwargs, a list uses *args.
    So 
    def my_function(a,b):
        pass 
    C/ list @ [1,2,3] won't work. C / my_function @ {'a':1,'b':2} will work.
    '''

    @__matmul__.register
    def __matmul__(self, other: typing.Callable) -> T:
        '''

        '''
        if self.col is None:
            return self.apply(other)
        if type(self.col) is dict:
            return other(**self.col)
        else:
            return other(*self.col)


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

        return CInst(gen(), unsafe=self._unsafe)

    @__lshift__.register
    def __lshift__(self, other: str):
        '''
        Act on each element of collection with function
        '''
        if self._unsafe & UnsafeType.Safe == UnsafeType.NotSafe:
            return self.__lshift__(CInst.conv_str_to_func(other))
        return self.__lshift__(other)

    def __rshift__(self, other):
        def gen():
            nonlocal other
            if isinstance(other,CInst):
                other=other.col
            for k in other:
                yield self.apply(k,simple=True)
        return   CInst( gen() , None)


    def __mod__(self, other):
        '''
        Applies paritial
        d= C / f % {'b': (lambda b:b*2)} @ (1,2,3)
        '''
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


            newsig = sig.replace(parameters=newparams.values())

            nf = partial(self.func, **nother)  # determined values

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

            fn = handle_currying() if self._unsafe & UnsafeType.Currying == UnsafeType.Currying else partial(
                self.func, **other)


        else:
            fn = partial(self.func, other)
        return CInst(fn, self.prev, unsafe=self._unsafe)

    def __getattr__(self,name):
        '''
        
        Runs function from outer scope.
        def find_computer(db , computer : str):
            print(db,computer)
        C.find_computer(db=Orig,computer=X.computer) / load_user @ ('a','b')
        '''
        import inspect
        locals= inspect.currentframe().f_back.f_locals
        if name in locals:
            func=locals[name]
        else:
            globals = inspect.currentframe().f_back.f_globals
            if name in globals:
                func = globals[name]
            else:
                return super().__getattr__(name)
                #raise AttributeError(f"no attribute {name}")


        def new_func(*args,**kw):
            return self / func / A(*args,**kw)
        return new_func

class CSimpInst(CInst):
    def __init__(self, unsafe=UnsafeType.NotSafe | UnsafeType.Currying):
        self.func = None
        self._unsafe = unsafe
        self.prev = None
        self.col = None
        self._a=None

    def __call__(self):
        raise NotImplementedError()
    def apply_int(self, origargs: CallWithArgs, args: typing.Any,simple=False) -> T:
        return args


C = CSimpInst()
CS = CSimpInst(unsafe=UnsafeType.Safe)
exp = Exp()
explist = ExpList()

