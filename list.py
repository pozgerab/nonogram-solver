from collections.abc import Callable, Sequence
from typing import Self, Literal, Iterator

type FieldValue = Literal[-1, 0, 1]  # Field value: -1, 0 or 1
type Consumer[C] = Callable[[C], None]  # Function with one arg and no return value
type Function[C, R] = Callable[[C, int], R]  # Function with one arg and one return value
type Condition[C] = Callable[[C], bool]  # Function with one arg and bool return value
type DoubleCondition[C] = Callable[
    [int, C, List[C]], bool]  # Function with an index and another arg and bool return value
type Merger[C] = Callable[[C, C], C]  # Function with two arguments and one return value
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

    def map[K](self, function: Function[T, K]) -> ListType[K]:
        """Maps through the list with a function"""
        return self.__class__([function(el, i) for i, el in enumerate(self)])

    def for_each(self, consumer: Consumer[T]):
        """Executes a consumer for each element of the list"""
        [consumer(el) for el in self]

    def filter(self, condition: Condition[T]) -> ListType[T]:
        """Filters the list with a condition"""
        return self.__class__([el for el in self if condition(el)])

    def filterWithIndex(self, condition: DoubleCondition[T]) -> ListType[T]:
        """Filters the list with a condition with index"""
        return self.__class__([el for i, el in enumerate(self) if condition(i, el, List[T](self))])

    def last_index_of(self, condition: Condition[T]):
        """Find the highest index where a condition is true"""
        return self.index(self.filter(condition)[-1])

    def first_index_of(self, condition: Condition[T]):
        """Find the lowest index where a condition is true"""
        return self.index(self.filter(condition)[0])

    def match_any(self, condition: Condition[T]):
        """If any of the element match the condition"""
        for el in self:
            if condition(el):
                return True
        return False

    def match_all(self, condition: Condition[T]):
        """If all the element match the condition"""
        for el in self:
            if not condition(el):
                return False
        return True

    def matchAllWithIndex(self, conditionWithIndex: DoubleCondition[T]):
        """If all the element match the condition. Iterates with index"""
        for i, el in enumerate(self):
            if not conditionWithIndex(i, el, self):
                return False
        return True

    def is_empty(self):
        return self.size() == 0

    def edges(self) -> Iterator[T]:
        if self.is_empty():
            return iter(())
        return iter((self[0], self[-1]))

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
    print(len(range(4,5)))
