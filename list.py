import inspect
from collections.abc import Callable
from itertools import count
from typing import Self, Literal, Iterator, overload

type FieldValue = Literal[-1, 0, 1]
type Consumer[C] = Callable[[C], None]
type Condition[C] = Callable[C, bool]
type Merger[C] = Callable[[C, C], C]
type Transformer[C] = Callable[[C], C]
type ListType[T] = List[T]


class List[T](list[T]):
    """List with more utility\n
    Mostly returns new lists and not references"""

    def __init__(self, *args):
        if isinstance(args, tuple):
            super().__init__(*args)
        else:
            super().__init__(args)

    @staticmethod
    def of(*args: T):
        return List[T](args)

    def __eq__(self, other):
        return super().__eq__(other)

    def flat(self):
        """Reduces a list of lists to one list of the elements of the nested lists. Only removes one level of nest"""
        return self.__class__(self.reduce(lambda l1, l2: l1 + l2))

    @overload
    def map[K](self, function: Callable[[T, int], K]): ...

    def map[K](self, function: Callable[[T], K]):
        param_count = len(inspect.signature(function).parameters)
        if param_count == 2:
            return self.__class__([function(el, i) for i, el in enumerate(self)])
        elif param_count == 1:
            return self.__class__([function(el) for el in self])
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    @overload
    def filter(self, condition: Condition[[T, int, Self]]):
        """Filters the list with a condition with index"""
        return self.__class__([el for i, el in enumerate(self) if condition(el, i, List[T](self))])

    def filter(self, condition: Condition[[T]]):
        """Filters the list with a condition"""
        return self.__class__([el for el in self if condition(el)])

    def last_index_of(self, condition: Condition[[T]]):
        """Find the highest index where a condition is true"""
        return self.index(self.filter(condition)[-1])

    def first_index_of(self, condition: Condition[[T]]):
        """Find the lowest index where a condition is true"""
        return self.index(self.filter(condition)[0])

    @overload
    def match_any(self, condition: Condition[[T, int]]): ...

    def match_any(self, condition: Condition[[T]]):
        param_count = len(inspect.signature(condition).parameters)
        if param_count == 2:
            return any(condition(i, el) for i, el in enumerate(self))
        elif param_count == 1:
            return any(condition(el) for el in self)
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    @overload
    def match_all(self, condition: Condition[[T, int]]): ...

    def match_all(self, condition: Condition[[T]]):
        param_count = len(inspect.signature(condition).parameters)
        if param_count == 2:
            return all(condition(i, el) for i, el in enumerate(self))
        elif param_count == 1:
            return all(condition(el) for el in self)
        else:
            raise TypeError("Callable must accept 1 or 2 arguments")

    def is_empty(self):
        return self.size() == 0

    def edges(self):
        if self.is_empty():
            return ()
        return self[0], self[-1]

    def split_index(self, index: int, end: int = None):
        return self.__class__(self[:index]), self.__class__(self[(end or index) + 1:])

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

    def size(self):
        """Length of the list"""
        return len(self)

    def reduce(self, reducer: Merger[T], order: Literal["normal", "reversed", "edges"] = "normal") -> T:
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

class IntList(List[int]):
    """List of ints"""

    def of(*args: int):
        return IntList(args)

    def sum(self):
        """Adds the ints together"""
        return self.reduce(lambda a, b: a+b)

    def are_adjacent(self):
        clone = IntList(self)
        clone.sort()
        for i, v in enumerate(clone):
            if i == clone.size() -1:
                continue
            if abs(v - clone[i + 1]) != 1:
                return False
        return True


if __name__ == "__main__":
    hf = IntList([0,1,2,3,4])
    del hf[0]
    print(hf)
    print(None or 0)
    print(list(IntList([0,1,2,3,4]).edges()))
