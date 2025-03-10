from __future__ import annotations

import inspect
import sys
from collections import deque
from collections.abc import Callable, Sequence
from itertools import count
from typing import Self, Literal, overload

type FieldValue = Literal[-1, 0, 1]
type ListType[T] = List[T]
type IntListType = ListType[int]

class List[T](list):
    """List with more utility\n
    Mostly returns new lists and not references"""

    @overload
    def __init__(self, *args, clause: Callable[[T], bool] = None): ...

    @overload
    def __init__(self, *args: Sequence[T], clause: Callable[[T], bool] = None): ...

    def __init__(self, *args, **kwargs):
        if isinstance(args, Sequence):
            super().__init__(*args)
        else: super().__init__(args)
        clause = kwargs.get("clause")
        self._clause = None
        if clause is not None and isinstance(clause, Callable):
            self._clause = clause

    @staticmethod
    def of(*args: T):
        return List[T](args)

    def __setitem__(self, key, value):
        if not self._clause or self._clause(value):
            super().__setitem__(key, value)

    def set_clause(self, clause: Callable[[T], bool]):
        self._clause = clause

    def flat(self):
        """Reduces a list of lists to one list of the elements of the nested lists. Only removes one level of nest"""
        return self.__class__(self.reduce(lambda l1, l2: l1 + l2))

    @staticmethod
    def wrap_list(func):
        def wrapper(self, *args, **kwargs):
            return self.__class__(func(self, *args, **kwargs))
        return wrapper

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(super().__getitem__(item))
        return super().__getitem__(item)

    @staticmethod
    def wrap_intlist(func):
        def wrapper(self, *args, **kwargs):
            return self.__class__(func(self, *args, **kwargs))
        return wrapper

    @staticmethod
    def wrap_with(inst: type):
        def _method(func):
            def wrapper(self, *args, **kwargs):
                return inst().__class__(func(self, *args, **kwargs))
            return wrapper
        return _method

    @overload
    def map[K](self, function: Callable[[T, int], K]) -> "List[K]": ...

    @overload
    def map[K](self, function: Callable[[T], K]) -> "List[K]": ...

    def map(self, function):
        param_count = len(inspect.signature(function).parameters)
        if param_count == 2:
            return self.__class__([function(el, i) for i, el in enumerate(self)])
        elif param_count == 1:
            return self.__class__([function(el) for el in self])
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    @overload
    def filter(self, condition: Callable[[T, int, Self], bool]) -> Self: ...

    @overload
    def filter(self, condition: Callable[[T], bool]) -> Self: ...

    def filter(self, condition):
        param_count = len(inspect.signature(condition).parameters)
        if param_count == 3:
            return self.__class__([el for i, el in enumerate(self) if condition(el, i, List(self))])
        elif param_count == 1:
            return self.__class__([el for el in self if condition(el)])
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    def last_index_of(self, condition: Callable[[T], bool]):
        """Find the highest index where a condition is true"""
        return self.index(self.filter(condition)[-1])

    def first_index_of(self, condition: Callable[[T], bool]):
        """Find the lowest index where a condition is true"""
        return self.index(self.filter(condition)[0])

    @overload
    def match_any(self, condition: Callable[[T, int], bool]) -> bool: ...

    @overload
    def match_any(self, condition: Callable[[T], bool]) -> bool: ...

    def match_any(self, condition):
        param_count = len(inspect.signature(condition).parameters)
        if param_count == 2:
            return any(condition(i, el) for i, el in enumerate(self))
        elif param_count == 1:
            return any(condition(el) for el in self)
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    @overload
    def match_all(self, condition: Callable[[T], bool]) -> bool: ...

    @overload
    def match_all(self, condition: Callable[[T, int], bool]) -> bool: ...

    def match_all(self, condition):
        param_count = len(inspect.signature(condition).parameters)
        if param_count == 2:
            return all(condition(i, el) for i, el in enumerate(self))
        elif param_count == 1:
            return all(condition(el) for el in self)
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    def is_empty(self):
        return self.size() == 0

    def edges(self) -> tuple[T, T]:
        if self.is_empty():
            raise ValueError("empty list")
        return self[0], self[-1]

    def split_index(self, index: int, end: int = None, remove_index: bool = True):
        return self[:index], self[(end or index) + remove_index:]

    def split(self, delimiter: T, limit: int = None):
        counter = count(1)
        copy = self.__class__(self)
        while copy:
            if delimiter not in copy:
                yield copy
                break
            index = copy.index(delimiter)
            if limit and next(counter) > limit:
                yield copy
                break
            a, copy = copy.split_index(index)
            yield a

    @overload
    def all_index(self, __value: T) -> List[IntList]: ...

    @overload
    def all_index(self, __value: Sequence[T]) -> List[IntList]: ...

    @wrap_list
    def all_index(self, __value):
        if __value not in self:
            return
        self_copy = self.__class__(self)
        _offset = 0
        while __value in self_copy:
            _index: int | Sequence[int] = self_copy.index(__value)

            _, self_copy = self_copy.split_index(_index[0]) if isinstance(__value, Sequence) else self_copy.split_index(_index)

            yield IntList([i + _offset for i in _index]) if isinstance(__value, Sequence) else _index + _offset

            _offset += (_index[0] if isinstance(__value, Sequence) else _index) + 1

    def size(self):
        return len(self)

    @overload
    def __contains__(self, item: T) -> bool: ...

    @overload
    def __contains__(self, item: Sequence[T]) -> bool: ...

    def __contains__(self, item):
        if isinstance(item, Sequence):
            if not item:
                return True
            if len(item) > len(self):
                return False

            if len(item) == 1:
                return super().__contains__(item[0])

            main_deque = deque(self)

            for _ in range(len(self) - len(item) + 1):
                if list(main_deque)[:len(item)] == item:
                    return True
                main_deque.popleft()

            return False
        else:
            return super().__contains__(item)

    @overload
    def index(self, __value: T, __start = 0, __stop = sys.maxsize) -> int: ...

    @overload
    def index(self, __value: Sequence[T], __start = 0, __stop = sys.maxsize) -> IntList: ...

    def index(self, __value, __start = 0, __stop = sys.maxsize):
        if isinstance(__value, Sequence):
            if __value not in self:
                raise ValueError(f"Not sublist of list: {__value}")
            start_index = super().index(__value[0], __start, __stop)
            return IntList([start_index + i for i in range(len(__value)) if start_index + i < __stop])
        else:
            return super().index(__value, __start, __stop)

    @overload
    def is_unique(self, __value: T) -> bool: ...

    @overload
    def is_unique(self, __value: Sequence[T]) -> bool: ...

    def is_unique(self, __value):
        return self.count(__value) == 1

    @overload
    def count(self, __value: T) -> int: ...

    @overload
    def count(self, __value: Sequence[T]) -> int: ...

    def count(self, __value):
        if isinstance(__value, Sequence):
            if not __value or len(__value) > len(self) or __value not in self:
                return 0
            _count = 0
            sub_len = len(__value)
            if sub_len == 1:
                return super().count(__value[0])
            main_deque = deque(self)

            for _ in range(len(self) - sub_len + 1):
                if list(main_deque)[:sub_len] == __value:
                    _count += 1
                main_deque.popleft()

            return _count
        else:
            return super().count(__value)

    @overload
    def ends_with(self, __value: T) -> bool: ...

    @overload
    def ends_with(self, __value: Sequence[T]) -> bool: ...

    def ends_with(self, __value):
        index = self.index(__value)
        if isinstance(index, list):
            return index[-1] == len(self) - 1
        else:
            return index == len(self) - 1

    @overload
    def starts_with(self, __value: T) -> bool: ...

    @overload
    def starts_with(self, __value: Sequence[T]) -> bool: ...

    def starts_with(self, __value):
        index = self.index(__value)
        if isinstance(index, list):
            return not index[0] # zero flipped -> true
        else:
            return not index

    def reduce(self, reducer: Callable[[T, T], T], order: Literal["normal", "reversed", "edges"] = "normal") -> T:
        """Reduces list. Return a new instance"""
        new = self.__class__(self)
        for i in range(self.size()):
            if new.size() < 2:
                return new[0]
            index = -2 if order == "reversed" else -1 if order == "edges" else 1
            new[-1 if order == "reversed" else 0] = reducer(new[-1 if order == "reversed" else 0], new[index])
            new.pop(index)
        return new[0]

    def get_diff_indexes(self, other: Self):
        return set([i for i, (a, b) in enumerate(zip(self, other)) if a != b])

    def is_all_exclusive(self):
        return self.size() == len(set(self))

    def n_lenght_adjacent_subsets(self, n: int) -> List[Self]:
        return self.__class__([self[i:i+n] for i in range(len(self) - n + 1)])

class IntList(List[int]):
    """List of ints"""

    @overload
    def __init__(self, *args: Sequence[int], clause: Callable[[int], bool] = None): ...

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def are_adjacent(self):
        clone = IntList(self)
        clone.sort()
        for i, v in enumerate(clone):
            if i == clone.size() -1:
                continue
            if abs(v - clone[i + 1]) != 1:
                return False
        return True

@overload
def join[K](delimiter: Sequence[K], elements: Sequence[Sequence[K]]) -> List[K]: ...

@overload
def join[K](delimiter: K, elements: Sequence[Sequence[K]]) -> List[K]: ...

def join(delimiter, elements):
    newl = List()
    delimiter = delimiter if isinstance(delimiter, Sequence) else [delimiter]
    for el in elements:
        newl += el
        newl += delimiter
    del newl[-len(delimiter):]
    return newl

if __name__ == "__main__":
    b = IntList([0,1,2,3])
    print(b.split_index(0, remove_index=False))
