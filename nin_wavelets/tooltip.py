from typing import Optional, List, Any, Callable, Tuple, Union, Dict, Iterator
from multiprocessing import Pool
import multiprocessing as mp
from os import cpu_count
from functools import partial, reduce
from itertools import starmap


def oneline_csv(*args: Tuple[Any]) -> str:
    result: str = ''
    comma: bool = False
    for arg in args:
        result = result + ',' + str(arg) if comma else str(arg)
        comma = True
    return result + '\n'


def not_none(x: Any) -> bool:
    if x is None:
        return False
    return True


def _parallel_run(seq: tuple) -> Any:
    result = seq[0](*seq[1:])
    return result


class Parallel:
    '''
    >>> para = Parallel(3)
    >>> para.append(test, 4)
    Parallel: test
    >>> para.append(test2, 2, 5)
    Parallel: test test2
    >>> para.run()
    [8, 10]
    '''
    def __init__(self, core: int = 2) -> None:
        self.args: list = []
        self.kwargs: list = []
        self.core = core

    def append(self, *args: tuple, **kwargs: dict) -> 'Parallel':
        self.args.append(args)
        self.kwargs.append(kwargs)
        return self

    def __repr__(self) -> str:
        result: str = 'Parallel:'
        for arg in self.args:
            result += ' ' + str(arg[0].__name__)
        return result

    def run(self) -> list:
        p = Pool(self.core)
        result = p.map_async(_parallel_run, self.args)
        return result.get()


def dict_map(func: Callable, dictionary: dict) -> Dict:
    '''
    >>> dict_map(test, {'hoge': 3, 'fuga': 4})
    {'hoge': 6, 'fuga': 8}
    '''
    data = {}
    for key, value in dictionary.items():
        data.update({key: func(value)})
    return data


def compose(*funcs: Callable) -> Callable:
    def wrap(arg: Any) -> Any:
        return reduce(lambda x, y: y(x), (arg,) + funcs)
    return wrap


class Sequence:
    '''
    >>> f = Sequence([2, 4, 6])
    >>> f[1]
    4
    '''
    def __init__(self, itr: Union[Iterator, List],
                 core: Optional[int] = 1) -> None:
        if core == 0:
            core = cpu_count()
        self.core = core
        if isinstance(itr, List):
            _itr = itr
        else:
            _itr = list(itr)
        self.len = len(_itr)
        self._num: int = 0
        self.data: Union[Iterator, List] = _itr
        self.data = self.data

    def map(self, func: Callable, **opt: Dict) -> 'Sequence':
        '''
        >>> Sequence([1, 2, 3], core=0).map(test).get()
        [2, 4, 6]
        >>> Sequence([1, 2, 3]).map(test).map(test)
        Sequence: [4, 8, 12]
        '''
        if len(opt.keys()) != 0:
            func = partial(func, **opt)
        data: List[Any]
        if self.core == 1:
            data = list(map(func, self.data))
            return Sequence(data, core=self.core)
        self.p = Pool(self.core)
        data = self.p.map_async(func, self.data).get()
        self.p.close()
        return Sequence(data, core=self.core)

    def starmap(self, func: Callable, **opt: Dict) -> 'Sequence':
        '''
        >>> Sequence(zip([1, 2, 3], [1, 2, 3])).starmap(test2)
        Sequence: [1, 4, 9]
        '''
        if len(opt.keys()) != 0:
            func = partial(func, **opt)
        data: List[Any]
        if self.core == 1:
            data = list(starmap(func, self.data))
            return Sequence(data, core=self.core)
        self.p = Pool(self.core)
        data = self.p.starmap(func, self.data)
        self.p.close()
        return Sequence(data, core=self.core)

    def __and__(self, itr: Iterator) -> 'Sequence':
        '''
        >>> Sequence([1]) & [4]
        Sequence: [1, 4]
        '''
        copy = Sequence(self.data)
        copy.data = list(self.data) + list(itr)
        return copy

    def __iter__(self) -> 'Sequence':
        self._num = 0
        return self

    def __next__(self) -> Any:
        if self._num == self.len:
            raise StopIteration()
        if isinstance(self.data, List):
            get = self.data[self._num]
            self._num += 1
            return get
        else:
            self.data = list(self.data)
            get = self.data[self._num]
            self._num += 1
            return get

    def get(self) -> Union[List, Iterator]:
        return self.data

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, key: int) -> Any:
        if isinstance(self.data, List):
            return self.data[key]
        else:
            return list(self.data)

    def __str__(self) -> str:
        '''
        >>> print(Sequence([3, 4, 5]))
        Sequence: [3, 4, 5]
        '''
        return 'Sequence: ' + str(self.data)

    def to_list(self) -> list:
        return list(self.data)

    def __repr__(self) -> str:
        return self.__str__()

    def reduce(self, func: Callable, **opt: Dict) -> Any:
        '''
        >>> from operator import add
        >>> Sequence([3, 4, 5]).reduce(add)
        12
        '''
        func = partial(func, **opt)
        return reduce(func, self.data)

    def filter(self, func: Callable, **opt: Dict) -> 'Sequence':
        '''
        >>> from operator import eq
        >>> Sequence([3, 4, 5]).filter(partial(eq, 4))
        Sequence: [4]
        '''
        func = partial(func, **opt)
        data = list(filter(func, self.data))
        return Sequence(data, core=self.core)


if __name__ == '__main__':
    def test(x: Any) -> Any:
        return x * 2

    def test2(x: Any, y: Any) -> Any:
        return x * y

    def test2_opt(x: Any, y: Any, mul: bool = False) -> Any:
        if mul:
            result = x * y
        else:
            result = x + y
        return result
    from doctest import testmod
    testmod()
    Sequence([1, 2, 3]).map(lambda x: x * 3).map(print)
