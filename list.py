from typing import Literal, Callable, Self

type FieldValue = Literal[-1, 0, 1] # Field value: -1, 0 or 1
type Consumer[C] = Callable[[C], None]  # Function with one arg and no return value
type Function[C,R] = Callable[[C, int], R]   # Function with one arg and one return value
type Condition[C] = Callable[[C], bool] # Function with one arg and bool return value
type ConditionWIndex[C] = Callable[[int, C, List[C]], bool]  # Function with an index and another arg and bool return value
type Reducer[C] = Callable[[C,C], C]  # Function with two arguments and one return value

class List[T](list[T]):
    """List with more utility\n
    Mostly returns new lists and not references"""

    def of(*args: T):
        return List[T](args)

    def plainer(self):
        '''Reduces a list of lists to one list of the elements of the nested lists. Only removes one level of nest'''
        return List(self.reduce(lambda l1, l2: l1 + l2)[0])
    
    def map(self, function: Function[T, any]):
        """Maps through the list with a function"""
        return List([function(el, i) for i, el in enumerate(self)])
    
    def forEach(self, consumer: Consumer[T]):
        '''Executes a consumer for each element of the list'''
        [consumer(el) for el in self]

    def filter(self, condition: Condition[T]):
        '''Filters the list with a condition'''
        passed = List[T]()
        [passed.append(el) if condition(el) else None for el in self]
        return passed
    
    def filterWithIndex(self, condition: ConditionWIndex[T]):
        '''Filters the list with a condition with index'''
        filtered = List[T]()
        [filtered.append(el) if condition(i, el, List[T](self)) else None for i, el in enumerate(self)]
        return filtered
    
    def findHighestIndex(self, condition: Condition[T]):
        '''Find the highest index where a condition is true'''
        return self.index(self.filter(condition)[-1])
    
    def findLowestIndex(self, condition: Condition[T]):
        '''Find the lowest index where a condition is true'''
        return self.index(self.filter(condition)[0])

    def contains(self, el: T):
        '''Whether an element is present'''
        return self.__contains__(el)
    
    def containsCondition(self, condition: Condition[T]):
        '''If a condition is present at least once'''
        return self.filter(condition).size() > 0
    
    def matchAny(self, condition: Condition[T]):
        '''If any of the element match the condition'''
        for el in self:
            if condition(el) == True:
                return True
        return False
    
    def matchAll(self, condition: Condition[T]):
        '''If all of the element match the condition'''
        for el in self:
            if condition(el) == False:
                return False
        return True
    
    def matchAllWithIndex(self, conditionWithIndex: ConditionWIndex[T]):
        """If all the element match the condition. Iterates with index"""
        for i, el in enumerate(self):
            if not conditionWithIndex(i, el, self):
                return False
        return True
    
    def countCondition(self, condition: Condition[T]):
        '''Count the element that match the condition'''
        return len(self.filter(condition))
    
    def isEmpty(self): return self.size() == 0
    
    def size(self):
        '''Lenght of the list'''
        return len(self)
    
    def reduce(self, reducer: Reducer[T], order: Literal["normal","reversed","twoend"] = "normal"):
        '''Reduces list. Return a new instance'''
        new = List(self)
        for i in range(self.size()):
            if new.size() < 2: return new
            new[-1 if order == "reversed" else 0] = reducer(new[-1 if order == "reversed" else 0], new[-2 if order == "reversed" else -1 if order == "twoend" else 1])
            new.pop(-2 if order == "reversed" else -1 if order == "twoend" else 1)
        return new
    
    def getDiffIndexes(self, other: Self):
        if self == other: return IntList()
        if self.size() != other.size(): raise Exception("Not equal size")
        diff = IntList()
        for i in range(self.size()):
            if self[i] != other[i]: diff.append(i)
        return diff
    
    def max(self, key: Callable[[T], int]):
        datamap = {key(i): i for i in self}
        return datamap[IntList(datamap.keys()).max()]
    
    def min(self, key: Callable[[T], int]):
        datamap = {key(i): i for i in self}
        return datamap[IntList(datamap.keys()).min()]
    
class IntList(List[int]):
    '''List of ints'''

    def of(*args: int):
        return IntList(args)
    
    def addUp(self):
        '''Adds the ints together'''
        v = 0
        for el in self:
            v += el
        return v
    
    def min(self) -> int:
        '''Get the lowest value of the list'''
        l = IntList(self)
        l.sort() 
        return l[0]
    def max(self) -> int:
        '''Get the highest value of the list'''
        l = IntList(self)
        l.sort()
        return l[-1]
    
class FieldValueList(List[Literal[-1,0,1]]):
    pass