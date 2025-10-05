import json
from collections.abc import Callable, Iterable
from itertools import repeat
from math import ceil, log10
from types import EllipsisType
from typing import Literal, Self, overload, Sequence

from portion import Interval, iterate, closed

from list import List, IntList, join

type Procedure = Callable[[GameLine], tuple[GameLine, set[int]]]
type Divider = Callable[[GameLine], list[tuple[range, GameLine]]]

type FieldValue = Literal[-1, 0, 1]

class Field:
    COLOR: FieldValue = 1
    EMPTY: FieldValue = 0
    EXCLUDE: FieldValue = -1

    @staticmethod
    def fieldset() -> set[FieldValue]:
        return {Field.COLOR, Field.EMPTY, Field.EXCLUDE}

def offset_set[T](og: set, offset: int) -> set[T]:
    return {e + offset for e in og}


class Line(List[Literal[-1, 0, 1]]):
    def __init__(self, __iterable: Iterable[FieldValue], **kwargs):
        super().__init__(__iterable, **kwargs)

    def __or__(self, other: Self):
        if isinstance(other, Line) and len(self) == len(other):
            return Line([self[i] or other[i] for i in range(self.size())])
        return None

    def get_index_if_value_is(self, index: int, value: FieldValue) -> int | None:
        return index if self[index] == value else None

    @staticmethod
    def indexes(func):
        """Indicates that the returned value are indexes (0 <= x < size)"""
        return func

    @staticmethod
    def values(func):
        """Indicates that the returned values are the field values (-1 | 0 | 1)"""
        return func

    @staticmethod
    def disallowx(n: FieldValue) -> bool:
        return n != -1

    @staticmethod
    def validate(func: Callable):
        def isvalid(self: Self, index: int):
            if not isinstance(index, int):
                print("HOOOOW")
            if not (0 <= index < self.size()):
                print("Out Of Bounds", f" {self}; {index}")
                return
            func(self, index)

        return isvalid

    def set_in_bound(self, index: int, value: FieldValue):
        if self.in_bound(index):
            self.set_field(index, value)

    def in_bound(self, index: int) -> bool:
        return 0 <= index < self.size()

    def get(self, index: int):
        return self[index]

    def set_field(self, index: int, value: FieldValue):
        self[index] = value

    def values_from_indexes(self, indexes: IntList) -> "Line":
        return Line([self[i] for i in indexes])

    @validate
    def fill(self, index: int):
        if self[index] != Field.EXCLUDE:
            self[index] = Field.COLOR
        else:
            print("Trying to color an impossible field")

    @validate
    def exclude(self, index: int):
        if self[index] != Field.COLOR:
            self[index] = Field.EXCLUDE
        else:
            print("Trying to exclude a colored field")

    @indexes
    def indexes_of_fields(self, fields: set[FieldValue]) -> IntList:
        """
        Gets the indexes of the given field values in the line. [-1,-1,0,1] + (0,1) -> [2,3]
        """
        return IntList(range(self.size())).filter(lambda index: self[index] in fields)

    @indexes
    def indexes_of_field(self, field: FieldValue) -> IntList:
        """
        Gets the indexes of a single field value in the line. [-1,-1,0,1] + 0 -> [2]
        """
        return self.indexes_of_fields({field})

    @indexes
    def group_adjacent_fields(self, fields: set[FieldValue]):
        """
        Groups multiple field indexes if they are connected. [0,0,-1,1,0,0] + (0,1) -> [[0,1], [3,4,5]]
        """
        list_: List[IndexList] = List()
        field_indexes = self.indexes_of_fields(fields)
        collector = IndexList(parent=self)
        for i, v in enumerate(field_indexes):
            if collector.size() == 0:
                collector.append(v)
                continue
            if field_indexes[i - 1] == v - 1:
                collector.append(v)
                continue
            list_.append(collector)
            collector = IndexList(parent=self)
            collector.append(v)
        if collector:
            list_.append(collector)
        return list_

    @indexes
    def group_adjacent_field(self, field: FieldValue):
        """
        Groups the field values if they are connected. [0,0,-1,1,0,0] + 0 -> [[0,1], [4,5]]
        """
        return self.group_adjacent_fields({field})

    @indexes
    def slice_by(self, field: FieldValue) -> List["IndexList"]:
        return self.group_adjacent_fields(Field.fieldset() - {field})

    @indexes
    def slice_by_x(self) -> "List[IndexList]":
        """ O-OX--O -> [0,1,2], [4,5,6]"""
        return self.slice_by(Field.EXCLUDE)

    @indexes
    def slice_by_x_contains_o(self) -> "List[IndexList]":
        return self.slice_by_x().filter(lambda group: Field.COLOR in group.values())

    @indexes
    def slice_by_x_contains_o_not_all_o(self) -> "List[IndexList]":
        return self.slice_by_x_contains_o().filter(lambda group: not group.values().match_all(lambda fv: fv == Field.COLOR))

    @indexes
    def slice_by_x_all_empty(self) -> "List[IndexList]":
        return self.slice_by_x().filter(lambda group: Field.COLOR not in group.values())

    @indexes
    def first_color_index_after_only_x(self) -> int | None:
        return self.get_index_if_value_is(self.slice_by_x()[0][0], Field.COLOR)

    @indexes
    def last_o_index_before_only_x(self) -> int | None:
        return self.get_index_if_value_is(self.slice_by_x()[-1][-1], Field.COLOR)

    @indexes
    def finished_tasks(self):
        finished_list: List[IndexList] = List()
        for group in self.slice_by(Field.EMPTY).filter(lambda _gr: Field.COLOR in _gr.values()):
            if tuple(group.values().edges()) == (Field.EXCLUDE, Field.EXCLUDE) and all(group.values()[1:-1]):
                finished_list.append(group)
            elif group[0] == 0 and group.values().edges()[-1] == Field.EXCLUDE and all(group.values()[:-1]):
                finished_list.append(group)
            elif group[-1] == range(self.size())[-1] and group.values().edges()[0] == Field.EXCLUDE and all(group.values()[1:]):
                finished_list.append(group)
        return finished_list

    def index_range(self) -> range:
        return range(self.size())

    def copy(self) -> "Line":
        return Line(self)

    @staticmethod
    def wrap_fit(func):
        def wrap(self, tasks):
            res = func(self, tasks)
            print(f'{tasks}{" DOES NOT" if not res else ""} fit in {self}')
            return res
        return wrap

    @wrap_fit
    def fits(self, tasks: IntList):
        tasks = IntList(tasks)
        current_index = 0
        while tasks:
            task = tasks[0]
            if current_index + task - 1 >= self.size(): return False
            part_index = IndexList(values = range(current_index, current_index + task), parent=self)
            part = part_index.values()
            if Field.EXCLUDE in part:
                current_index += part.index(Field.EXCLUDE) + 1
                continue
            if self.in_bound(part_index[0] - 1) and self[part_index[0] - 1] == Field.COLOR:
                current_index += 1
                continue
            if self.in_bound(part_index[-1] + 1) and self[part_index[-1] + 1] == Field.COLOR:
                current_index += 1
                continue
            current_index = part_index[-1] + 2
            del tasks[0]
        return True

    def set_slice(self, index_range: range, line: "Line"):
        for i, slice_index in enumerate(index_range):
            self.set_field(slice_index, line[i])

class IndexList(IntList):
    def __init__(self, parent: Line, values:Sequence=()):
        super().__init__(values)
        self.parent = parent

    def values(self):
        return self.parent.values_from_indexes(self)

    def offset(self):
        if self.is_empty():
            raise IndexError("IndexList is empty")
        return self[0]

    def range(self):
        return range(self[0], self[-1] + 1)

    def yieldable(self, tasks: IntList, clause: Callable[[FieldValue], bool] | None = None):
        """
        Creates a tuple whose structure is correct to be yielded to a @divider method
        :param clause: line clause
        :param tasks: tasks that will belong to this divided slice
        :return: a yieldable object to @divider methods
        """
        return self.range(), GameLine(self.values() if not clause else Line(self.values(), clause = clause), tasks)

class ProcedureLib:
    PROCEDURES: dict[str, Procedure] = dict()
    DIVIDERS: dict[str, Divider] = dict()
    @staticmethod
    def get_names() -> set[str]:
        return set(ProcedureLib.PROCEDURES.keys())

    @staticmethod
    def procedures():
        return set(ProcedureLib.PROCEDURES.values())

    @staticmethod
    def dividers():
        return set(ProcedureLib.DIVIDERS.values())


class GameLine(tuple[Line, IntList]):

    def __init__(self, line: Line, task: IntList):
        self.__line, self.task = line, IntList(task)

    @property
    def line(self):
        return self.__line

    def __new__(cls, line: Line, task: IntList):
        instance = super().__new__(cls, (line, task))
        return instance

    def __repr__(self):
        return ",".join(str(e) for e in self.task) + "|".join(["▁▁" if self.line[i] == 0 else "██" if self.line[i] == 1 else "░░" for i in range(self.line.size())])

    @staticmethod
    def procedure(name: str):
        def decorator(procedure: "Callable[[GameLine], None]"):
            def wrapper(game_line) -> "tuple[GameLine, set[int]]":
                if Field.EMPTY not in game_line.line:
                    print(f"Proc::'{name}': Line already finished: {game_line}")
                    return game_line, set()
                before = Line(game_line.line)
                procedure(game_line)
                diffs = before.get_diff_indexes(game_line.line)
                changed = len(diffs) != 0
                print(f"Procedure '{name}' executed. {("No " if not changed else '') + "Diff"} {diffs if changed else ''}")
                return game_line, diffs

            ProcedureLib.PROCEDURES[name] = wrapper
            return wrapper

        return decorator

    @staticmethod
    def divider(name: str):
        def decorator(divider: Divider):
            def wrapper(game_line) -> "list[tuple[range, GameLine]]":
                more_than_expected = False
                divided: List[tuple[range, GameLine]] = List(divider(game_line) or [])
                divided = divided.filter(lambda div: len(div[0]) < game_line.line.size())
                unique_ranges = set(divided.map(lambda div: div[0]))    # keep unique ranges

                if len(unique_ranges) != len(divided):
                    more_than_expected = True
                    print(f'\033[31mDivider {name} has produced multiple of the same ranges, FIX IT\033[0m')
                    # TODO error below
                    #divided = List([divided[divided.first_index_of(lambda div: div[0] == unique_range)] for unique_range in unique_ranges])
                print(f"Divider '{name}' executed\n\tdivided = {divided}")
                if more_than_expected:
                    input("Just so you know")
                return divided
            ProcedureLib.DIVIDERS[name] = wrapper
            return wrapper

        return decorator

    def possible_finished_groups(self) -> List[IndexList]:
        return self.line.group_adjacent_field(Field.COLOR).filter(lambda group: self.task.match_any(lambda _t: _t == group.size()))

    @divider(name = "splits if a task is certainly finished")
    def split(self):
        finished_groups = self.line.finished_tasks() # blocks that are sorrounded by X and don't contain empty fields
        tasks_copy = IntList(self.task)
        line_copy = Line(self.line)
        range_copy = range(self.line.size())

        finished_tasks = list[list[int]]()
        for f_group in finished_groups:     # Separates the blocks to only individual colored blocks with its lengths (= tasks)
            adjacent_sizes = []
            for adj in Line(f_group.values()).group_adjacent_field(Field.COLOR):
                adjacent_sizes.append(adj.size())
            finished_tasks.append(adjacent_sizes)

        finished_blocks = list(zip(finished_groups, finished_tasks))    # [(blocks, tasks)...]

        while finished_blocks:      # while there are unchecked finished blocks
            first_group, first_tasks = finished_blocks[0]
            if tasks_copy.starts_with(first_tasks) and tasks_copy.is_unique(first_tasks):   # if finsihed tasks are the first out of all tasks and unique
                _, line_copy = line_copy.split_index(first_group[-1])                       # remove before and group part
                for i in range(len(first_tasks)):                               # remove tasks
                    del tasks_copy[0]
                range_copy = range(first_group[-1] + 1, range_copy[-1] + 1)       # indicates the remaining index range of the whole line
                del finished_blocks[0]
                continue            # and advance to the remaining finished groups

            last_group, last_tasks = finished_blocks[-1]
            if tasks_copy.ends_with(last_tasks) and tasks_copy.is_unique(last_tasks):   # same as the previous just from the end
                line_copy, _ = line_copy.split_index(last_group[0] - range_copy[0])     # remove after and group part
                for i in range(len(last_tasks)):
                    del tasks_copy[-1]
                range_copy = range(range_copy[0], last_group[0])
                del finished_blocks[-1]
                continue

            for iter_index, (group, tasks) in enumerate(finished_blocks):   # loop through finished block that are in the middle or not unique

                if tasks_copy.is_unique(tasks):         # if tasks are unique, pass the before part to be solved, and delete that and the finished groups themselves.
                    tasks_indexes = tasks_copy.index(tasks)
                    sep_range = range(range_copy[0], group[0])
                    sep_line, line_copy = line_copy.split_index(*group.edges())
                    sep_task, tasks_copy = tasks_copy.split_index(*tasks_indexes.edges())
                    range_copy = range(group[-1] + 1, range_copy[-1] + 1)
                    del finished_blocks[iter_index]
                    yield sep_range, GameLine(Line(sep_line), IntList(sep_task))
                    continue

                # if not unique
                possible_indexes: List[IntList] = tasks_copy.all_index(tasks)
                passed_indexes: list[IntList] = []
                before_line, after_line = line_copy.split_index(*(index - range_copy[0] for index in group.edges())) # subtract the offset
                for indexes in possible_indexes:
                    before_tasks, after_tasks = tasks_copy.split_index(*indexes.edges())
                    before_min, after_min = sum(before_tasks, max(len(before_tasks) - 1, 0)), sum(after_tasks, max(len(after_tasks) - 1, 0))
                    if before_line.size() >= before_min and after_line.size() >= after_min:
                        passed_indexes.append(indexes)
                if len(passed_indexes) == 1:
                    found_indexes = passed_indexes[0]
                    sep_range = range(range_copy[0], group[0])
                    sep_line, line_copy = line_copy.split_index(*(index - range_copy[0] for index in group.edges()))
                    sep_task, tasks_copy = tasks_copy.split_index(*found_indexes.edges())
                    range_copy = range(group[-1] + 1, self.line.size())
                    yield sep_range, GameLine(Line(sep_line), IntList(sep_task))
                del finished_blocks[iter_index]

        if tasks_copy and line_copy and self.line.size() != line_copy.size():
            yield range_copy, GameLine(line_copy, tasks_copy)

    @divider(name = "if started == tasks")
    def splitifstartedeqtask(self):
        groups = self.line.slice_by_x_contains_o()
        if groups.size() == self.task.size():
            for i, group in enumerate(groups):
                yield range(group[0], group[-1] + 1), GameLine(group.values(), IntList([self.task[i]]))

    @divider(name = "remove start-end excludes")
    def trim(self):
        if Field.EXCLUDE not in self.line.edges(): return
        yield range(*self.line.slice_by_x().flat().edges()), GameLine(Line(join(-1, self.line.slice_by_x().map(lambda _gr: _gr.values()))), self.task)

    @procedure(name = "connect certain connectibility")
    @divider(name = "separate unconnectable")
    def seperateunconnectable(self):
        color_groups = self.line.group_adjacent_field(Field.COLOR)
        if self.task.size() >= color_groups.size():
            return

        excess_group_am = color_groups.size() - self.task.size()
        subs_max_length = excess_group_am + 1
        connections_needed = excess_group_am

        passed_groups = List()

        for length in range(subs_max_length, 1, -1):
            print(f'{length=}')

            subsets = color_groups.n_lenght_adjacent_subsets(length)
            poss_task_indexes = [set[int]() for _ in subsets]
            [poss_task_indexes[i].add(i) for i in range(1, subsets.size()) if i < self.task.size()]
            [poss_task_indexes[-(i+1)].add(-(i + 1)) for i in range(1, subsets.size()) if i + 1 <= self.task.size()]

            print(f'{subsets=}')
            print(f'{poss_task_indexes=}')

            for possible_task_index_arr, (first, *mid, last) in zip(poss_task_indexes, subsets):

                dist = last[-1] - first[0] + 1
                print(f'{dist=}')
                for task_index in possible_task_index_arr:
                    print(f'{dist <= self.task[task_index]=}')
                    if dist <= self.task[task_index]:
                        passed_groups.append((task_index, first, last))
                        connection_amount = 2 + len(mid) - 1
                        connections_needed -= connection_amount
                        if connections_needed < 0:
                            return

        if connections_needed == 0:
            for i, first, last in passed_groups:
                for _f in range(first[-1] + 1, last[0]):
                    self.line.fill(_f)

                self.surroundIfBiggest()

                bef_task, aft_task = self.task.split_index(i)
                bef_line, aft_line = self.line.split_index(first[0], last[-1])
                yield range(0, first[0]), GameLine(Line(bef_line, clause = Line.disallowx), bef_task)
                yield range(last[-1] + 1, self.line.size()), GameLine(Line(aft_line, clause = Line.disallowx), aft_task)

    @divider(name = "first-last only fits in first/last")
    def certaintaskpositions(self):
        x_slices = self.line.slice_by_x()
        if x_slices.size() == 1:
            return

        #   DISCLAIMER for me       This divider is just so "technical", please don't modify this thinking through all of it again
        #                           Also, a recursive function would be more pratical maybe, could implement in the future

        available_tasks = IntList(self.task)
        start_checked, end_checked = False, False
        splt_i = 0 # index of the group the line will be sliced after
        max_split_index = x_slices.size() - 2
        from_start = True # whether start from the start(True) or the end(False)

        while available_tasks:

            if x_slices.size() == 1: # everything other than this has found a spot
                break

            if all([start_checked, end_checked]): # if checked without any found
                splt_i += 1                       # increase the group index that the slicing is done after, basically increase the amout of groups to be checked at once
                from_start = True
                if splt_i > max_split_index:      # if it goes over, couldn't find any certain task spots :(
                    break

            print(f'{splt_i=}')

            if from_start:

                start_checked = True
                for end in range(1, available_tasks.size() - 1):

                    print(f'next {x_slices=}')

                    tasks_first, tasks_rest = available_tasks.split_index(end, remove_index=False)              # just the sliced tasks
                    tasks_first_intr, tasks_rest_intr = available_tasks.split_index_intersect(end - 1, end)     # the sliced tasks but with 2 intersecting elements

                    # separate the line by the split index
                    first_slice, rest_slice = IntList(range(self.line.size())).split_index(x_slices[splt_i][-1]+1, x_slices[splt_i + 1][0], remove_index=False)
                    first_slice = IndexList(self.line, first_slice) # just link these to our actual line
                    rest_slice = IndexList(self.line, rest_slice)

                    first_slice_conditions = first_slice.values().fits(tasks_first), not first_slice.values().fits(tasks_first_intr) # tasks fit, but one more does not
                    rest_slice_conditions = rest_slice.values().fits(tasks_rest), not rest_slice.values().fits(tasks_rest_intr)      # rest tasks fit, but one more does not

                    if all(first_slice_conditions + rest_slice_conditions): # if all conditions met, we can state that the tasks can only belong to that certain groups
                        yield first_slice.yieldable(tasks_first)

                        for _ in repeat(None, tasks_first.size()): # remove used tasks
                            del available_tasks[0]

                        for _ in repeat(None, splt_i + 1): # remove used slices
                            del x_slices[0]
                            print(f'after del {x_slices=}')

                        start_checked = False # reset so the rest can be checked again
                        break                 # break from loop and check again for rest

                from_start = not from_start    # switch directions

            else:
                #   !!! Read the "from start" block, basically the same but the other way around
                end_checked = True
                for start in range(available_tasks.size()-1, 1, -1):
                    tasks_rest, tasks_last = available_tasks.split_index(start, remove_index=False)
                    tasks_rest_intr, tasks_last_intr = available_tasks.split_index_intersect(start-1, start)

                    print(f'{x_slices=}')

                    rest_slice, last_slice = IntList(range(self.line.size())).split_index(x_slices[-(2 + splt_i)][-1]+1, x_slices[-(1 + splt_i)][0], remove_index=False)
                    rest_slice = IndexList(self.line, rest_slice)
                    last_slice = IndexList(self.line, last_slice)

                    rest_slice_conditions = rest_slice.values().fits(tasks_rest), not rest_slice.values().fits(tasks_rest_intr)
                    last_slice_conditions = last_slice.values().fits(tasks_last), not last_slice.values().fits(tasks_last_intr)

                    if all(last_slice_conditions + rest_slice_conditions):
                        yield last_slice.yieldable(tasks_last)

                        for _ in repeat(None, tasks_last.size()):
                            del available_tasks[-1]

                        for _ in repeat(None, splt_i + 1):
                            del x_slices[-1]

                        end_checked = False
                        break

                from_start = not from_start    # switch directions

            if available_tasks:         # if some tasks has not found spots, just yield the remaining groups in one as well as the tasks,
                                        #   as it is certain that those will be located there, just with less precision
                _range = range(x_slices[0][0], x_slices[-1][-1] + 1)
                yield _range, GameLine(self.line[_range], available_tasks)

    @divider(name = "if individual tasks cant fit with before tasks")
    def individualtask(self):
        x_slices = self.line.slice_by_x()
        if x_slices.size() == 1:
            return

        finished_indexes = List[int]()
        finished_tasks = List[int]()

        working_tasks = IntList(self.task)

        for _sl_i, _slice in enumerate(x_slices):      # collect finished tasks indexes and sizes
            if Field.EMPTY not in _slice.values():
                finished_indexes.append(_sl_i)
                finished_tasks.append(_slice.size())

        for _, f_task in zip(finished_indexes, finished_tasks):
            if working_tasks.is_unique(f_task):                 # if task is unique, remove the finished task and don't take the group into calculation
                del working_tasks[working_tasks.index(f_task)]
                continue

            indexes = self.task.all_index(f_task)      # if not unique, search possible indexes
            if indexes.are_adjacent():                 # if all possible indexes are next to each other, it does not matter where we remove them
                del working_tasks[indexes[0]]          # if not next to each other just ignore bc we do not know which task is fulfilled

        not_done_slices = x_slices.filter(lambda _g: 0 in _g.values())  # x-slices that are not finished

        yields = []     # don't yield right away bc merging is needed

        for i, task in enumerate(working_tasks): # iterate through all tasks with their indexes, we are only inspecting singular tasks in this @divider

            passed_groups = List()
            for g_i, group in enumerate(not_done_slices):               # iterate through all unsolved groups
                is_last = g_i == not_done_slices.size() - 1             # if group is the last out of all unfinished groups

                before_task, after_task = working_tasks.split_index(i)  # splits the unfinished tasks to the ones before and after the iterated task

                if g_i == 0:    # if the group is the first out of all unfinished
                    if sum(before_task, task + before_task.size()) <= group.size():     # if before tasks and itself fit in the group
                        passed_groups.append((g_i, before_task + [task]))               # potential solution ?
                elif is_last:
                    print(f'{sum(after_task, task + after_task.size()) <= group.size()=}')
                    if sum(after_task, task + after_task.size()) <= group.size():   # if after tasks and itself fit in the group
                        passed_groups.append((g_i, [task] + after_task))            # potential solution ?
                elif task <= group.size():      # if in middle of groups and task fits
                    passed_groups.append((g_i, [task])) # potential solution ?
            print(f'{passed_groups.size()=}')
            if passed_groups.size() == 1:       # if only one groups produced a potential solution
                index, tasks = passed_groups[0]
                group = not_done_slices[index]
                # do not remove this group from the iteration because it is possible for multiple tasks to only fit in that group, this is why merging is needed

                # !!! Clause is neccessary bc not only these tasks can be in this group exclusively, so basically soft-solve
                yields.append(group.yieldable(IntList([task]), clause = Line.disallowx)) # add the group and task to be  (merged then) yielded

        ranges = dict()
        for _range, line in yields:         # If two task is for the same range, merge the tasks
            if _range not in ranges:
                ranges[_range] = line
            else:
                merged = ranges[_range].task + line.task
                ranges[_range] = GameLine(ranges[_range].line, merged)

        for __range, __line in ranges.items(): # Clase is kept here, I checked
            yield __range, __line

    def get_too_small_groups(self):
        return self.line.slice_by_x_all_empty().filter(lambda group: self.task.match_all(lambda task: task > group.size()))

    def is_solved_diffs(self):
        solved = self.task == self.line.group_adjacent_field(Field.COLOR).map(lambda gr, i: gr.size())
        to_be_excluded = set()
        if solved:
            to_be_excluded = set(self.line.indexes_of_field(Field.EMPTY))
            for ind in to_be_excluded:
                self.line.exclude(ind)
        return solved, to_be_excluded

    def is_solved(self):
        return self.is_solved_diffs()[0]

    @procedure(name = "first or last group is smaller than first or last task")
    def exclfirstsmall(self):
        while self.line.slice_by_x()[0].size() < self.task[0]:
            for field in self.line.slice_by_x()[0]:
                self.line.exclude(field)

        while self.line.slice_by_x()[-1].size() < self.task[-1]:
            for field in self.line.slice_by_x()[-1]:
                self.line.exclude(field)

    @procedure("if empty group = task amount && group size = task -> color all")
    def procZero(self):
        groups = self.line.slice_by_x()
        if groups.size() == self.task.size() and groups.match_all(
                lambda index, group: group.size() == self.task[index]):
            indexes = groups.flat()
            for i in range(self.line.size()):
                if i in indexes:
                    self.line[i] = Field.COLOR

    @procedure("remove too small groups")
    def removeTooSmall(self):
        for small_g in self.get_too_small_groups():
            for i in small_g:
                self.line.exclude(i)

    @procedure("default starting interval solve")
    def procedureOne(self):

        taskSize = self.task.size()

        iterator = 0
        taskMap = {i: {"start": 0, "end": 0} for i in range(
            taskSize)}  # Indicate starts and ends with value (placeholder 0)
        intervalMap: dict[int, Interval] = {}  # every task will have a certain range
        for index, task in enumerate(self.task):  # indicate min end index of task
            taskMap[index]["end"] = iterator + task - 1
            iterator += task + 1
        iterator = self.line.last_o_index_before_only_x() or self.line.size() - 1
        for index, task in enumerate(self.task):  # indicate max starting index of task
            taskMap[taskSize - index - 1]["start"] = iterator - self.task[-index - 1] + 1
            iterator -= self.task[-index - 1] + 1
        for index, v in taskMap.items():  # make interval out of end-start values
            if v["start"] > self.line.size() - 1 or v["end"] > self.line.size() - 1: continue
            intervalMap[index] = closed(v["start"], v["end"])

        for p in intervalMap.values():  # the intersection will be certainly colored
            for field in iterate(p, 1):
                self.line.fill(field)

    @procedure("if color on side -> its always the start")
    def procIfSide(self):
        i, j = 0, 0
        first_c_index: int | None = self.line.first_color_index_after_only_x()
        if first_c_index is not None:
            for i in range(first_c_index, first_c_index + self.task[0]):
                self.line.fill(i)
            self.line.set_in_bound(i + 1, Field.EXCLUDE)

        last_c_index: int | None = self.line.last_o_index_before_only_x()
        if last_c_index is not None:
            for j in range(last_c_index - self.task[-1] + 1, last_c_index):
                self.line.fill(j)
            self.line.set_in_bound(last_c_index - self.task[-1], Field.EXCLUDE)

    @procedure("if near start or end then continue the sequence")
    def continuestartorend(self):
        if Field.COLOR not in self.line:
            return

        first_task, last_task = self.task.edges()
        first_color, last_color = self.line.indexes_of_field(Field.COLOR).edges()
        if first_task - 1 > first_color:
            first_continue_range = range(first_color + 1, first_task)
            for i in first_continue_range:
                self.line.fill(i)
            if len(first_continue_range) == first_task - 1:
                self.line.set_in_bound(first_task, Field.EXCLUDE)
        if self.line.size() - last_task < last_color:
            last_continue_range = range(self.line.size() - last_task, last_color)
            for j in last_continue_range:
                self.line.fill(j)
            if len(last_continue_range) == last_task - 1:
                self.line.set_in_bound(self.line.size() - last_task - 1, Field.EXCLUDE)

    @procedure("if biggest task can be only in one group")
    def procIfOneIfBiggerThanAny(self):
        if self.task.size() == 1 or Field.COLOR not in self.line:
            return
        started = self.line.slice_by_x()

        biggestTask = max(self.task)
        big_gs = started.filter(lambda g: g.size() >= biggestTask)
        if big_gs.size() == 1:
            biggest_g = big_gs[0]
            biggest_g.yieldable(IntList([biggestTask]))[1].procedureOne()

    @procedure("if first or last group contains color and is the size of the first task -> color")
    def procIfCertainG(self):
        groups = self.line.slice_by_x()
        if Field.COLOR in groups[0].values() and groups[0].size() == self.task[0]:
            for i in groups[0]:
                self.line.fill(i)
        if Field.COLOR in groups[-1].values() and groups[-1].size() == self.task[-1]:
            for i in groups[-1]:
                self.line.fill(i)

    @procedure("if task == started group -> color certain, exclude impossible")
    def procgroupeqtask(self):
        started_groups = self.line.slice_by_x_contains_o()
        if started_groups.size() == self.task.size():
            # Exclude every group that is not started
            [self.line.exclude(field) for field in range(self.line.size()) if field not in started_groups.flat()]

            for group, task in zip(started_groups, self.task):
                if group.size() == task:
                    for field in group:
                        self.line.fill(field)
                elif group.values().count(Field.COLOR) == task:
                    [self.line.exclude(field) for field in group if self.line != Field.COLOR]
                else:
                    small_line = GameLine(group.values(), IntList([task]))
                    small_line.procedureOne()
                    for i, v in zip(group, small_line.line):
                        self.line[i] = v

    @procedure("if an adjacent colored list is same size as biggest task -> surround with x")
    def surroundIfBiggest(self):
        color_groups = self.line.group_adjacent_field(Field.COLOR)
        biggest_task = max(self.task)
        for group in color_groups:
            if group.size() == biggest_task:
                for edge in (group[0] - 1, group[-1] + 1): #    Two surrounding fields
                    if self.line.size() > edge >= 0 == self.line[edge]:
                        self.line.exclude(edge)

    @procedure("if one task and at least one colored field -> exclude unreachable")
    def excludeunreachable(self):
        if self.task.size() != 1 or Field.COLOR not in self.line:
            return
        task = self.task[0]

        colored_fields = self.line.indexes_of_field(Field.COLOR)
        first_colored, last_colored = colored_fields[0], colored_fields[-1]
        for i in range(first_colored + 1, last_colored): # connect fields
            self.line.fill(i)

        excluded_from_start = range(0, max(last_colored - task + 1, 0))
        excluded_from_end = range(min(first_colored + task, self.line.size()), self.line.size())
        excluded = set().union(excluded_from_end, excluded_from_start)

        for exc_iter in excluded:
            self.line.exclude(exc_iter)

    @procedure("if exclusively biggest task is on edge and a color block can only be that -> exclude towards edge")
    def exclusivebiggestexclude(self):
        if Field.COLOR not in self.line:
            return
        if max(self.task) == self.task[0] and self.task.count(self.task[0]) == 1:
            task = self.task[0]
            coloredgroups = self.line.group_adjacent_field(Field.COLOR)
            biggestGroups = coloredgroups.filter(lambda _group: IntList(self.task[1::]).match_all(lambda _task: _group.size() > _task))
            if biggestGroups.size() == 1:
                group = biggestGroups[0]
                excluded = range(0, max(group[-1] - task + 1, 0))
                for i in excluded:
                    self.line.exclude(i)
        elif max(self.task) == self.task[-1] and self.task.count(self.task[-1]) == 1:
            task = self.task[-1]
            coloredgroups = self.line.group_adjacent_field(Field.COLOR)
            biggestGroups = coloredgroups.filter(lambda _group: IntList(self.task[:-1:]).match_all(lambda _task: _group.size() > _task))
            if biggestGroups.size() == 1:
                group = biggestGroups[0]
                excluded = range(min(group[0] + task, self.line.size()), self.line.size())
                for i in excluded:
                    self.line.exclude(i)

    def solve(self) -> set[int]:
        all_diffs = set[int]()
        for start_proc in ProcedureLib.procedures():
            starting_game_line, starting_proc_diffs = start_proc(self)
            all_diffs.update(starting_proc_diffs)

        def traverse_divide(index_range: range, line_part: GameLine, diffs: set[int]):
            solved, exclusion_diffs = line_part.is_solved_diffs()
            if solved:
                self.line.set_slice(index_range, line_part.line)
                diffs.update(exclusion_diffs)
                return
            for divider in ProcedureLib.dividers():
                sub_parts = divider(line_part)
                if len(sub_parts) == 0:
                    for proc in ProcedureLib.procedures():
                        game_slice, proc_diffs = proc(line_part)
                        if Field.EMPTY not in game_slice.line:
                            break
                        if len(proc_diffs) != 0:
                            self.line.set_slice(index_range, game_slice.line)
                            diffs.update(offset_set(proc_diffs, index_range.start)) # offset diff
                else:
                    for sub_range, sub_part in sub_parts:
                        sub_solved, sub_exclusion_diffs = sub_part.is_solved_diffs()
                        if sub_solved and len(sub_exclusion_diffs) != 0:
                            self.line.set_slice(index_range, sub_part.line)
                            diffs.update(offset_set(sub_exclusion_diffs, index_range.start))
                            continue
                        start = sub_range.start + index_range.start # offset with prev sub part
                        end = sub_range.stop + index_range.start
                        traverse_divide(range(start, end), sub_part, diffs)

        traverse_divide(self.line.index_range(), self, all_diffs)
        _, end_check_diffs = self.is_solved_diffs()
        all_diffs.update(end_check_diffs)
        return all_diffs

class Board(List[list[FieldValue]]):

    def __init__(self, **kwargs):
        super().__init__()
        self.rowTask: List[IntList] = List(kwargs.get("rowTask"))
        self.colTask: List[IntList] = List(kwargs.get("colTask"))
        self.height = len(self.colTask)
        self.width = len(self.rowTask)
        [self.append([Field.EMPTY for _ in range(self.width)]) for _ in range(self.height)]

        self.latest_changes: set[type[int, int]] = set()

    def get(self, x: int, y: int):
        return self[x][y]

    def __getitem__(self, index: int | tuple[EllipsisType, int]):
        if isinstance(index, int):
            return super().__getitem__(index)
        if isinstance(index, tuple):
            return Line([super().__getitem__(c)[index[1]] for c in range(self.width)])
        raise TypeError("wrong arguments")

    def __repr__(self) -> str:
        return f'Board[{self.height}][{self.width}]'

    def print(self):
        longest_row_task_length = max(self.colTask.map(lambda tasks: len(tasks)))

        space_for_task = ceil(log10(max(self.rowTask.flat()))) + 1 # how many digits + 1
        max_col_task_len = max(self.colTask.map(lambda tasks: len(tasks)))

        horizontal_task_padding = space_for_task * (max_col_task_len + 2) # + 2 bc of index display
        vertical_task_space = space_for_task * max_col_task_len
        horizontal_padding_str = " " * horizontal_task_padding

        print("Board[][]::\n")
        print(horizontal_padding_str + "".join([f'{i:<3}' for i in range(self.width)]))

        for task_line_idx in range(longest_row_task_length):
            line_task_strings = self.rowTask.map(lambda tasks: str(tasks[len(tasks) - longest_row_task_length + task_line_idx]) if longest_row_task_length - task_line_idx <= len(tasks) else "")
            print(horizontal_padding_str + "".join([f'{task:<3}' for task in line_task_strings]))
        for i in range(self.height):

            for col_task_idx in range(max_col_task_len):
                pass

            print(f'{i:<{space_for_task * 2}}{"".join([f'{task:>{space_for_task}}' for task in self.colTask[i]]):>{vertical_task_space}}'
                  +"|".join([("\033[31m" if (j,i) in self.latest_changes else "") + ("▁▁" if self[i][j] == 0 else "██" if self[i][j] == 1 else "░░") + ("\033[0m" if (j,i) in self.latest_changes else "") for j in range(self.width)]),
                  sep="")

    def get_column(self, index: int) -> GameLine:
        return GameLine(self[..., index], self.rowTask[index])

    def get_row(self, index: int) -> GameLine:
        return GameLine(Line(self[index]), self.colTask[index])

    def set_column(self, index: int, line: Line):
        for c in range(self.width):
            self[c][index] = line[c]

    def set_row(self, index: int, line: Line):
        self[index] = line

    def get_field(self, column=0, row=0):
        return self[column][row]

    def solve_line(self, index: int, getter: Callable[[int], GameLine], setter: Callable[[int, Line], ...]) -> set[int]:
        all_diffs = set[int]()
        game_line = getter(index)
        game_line.solve()
        setter(index, game_line.line)
        return all_diffs

    def solve_row(self, index: int) -> set[int]:
        return self.solve_line(index, getter = self.get_row, setter = self.set_row)

    def solve_column(self, index: int) -> set[int]:
        return self.solve_line(index, getter = self.get_column, setter = self.set_column)

    def row_proc(self, index: int, proc: Procedure) -> set[int]:
        """Applies the given procedure to a row at a given index. Returns the changed indexes"""
        game_line, diff = proc(self.get_row(index))
        self.set_row(index, game_line.line)
        return diff

    def col_proc(self, index: int, proc: Procedure) -> set[int]:
        """Applies the given procedure to a column at a given index. Returns the changed indexes"""
        game_line, diff = proc(self.get_column(index))
        self.set_column(index, game_line.line)
        return diff

    def is_solved(self):
        for i in range(self.height):
            if not self.get_row(i).is_solved():
                return False
        return True

    def is_empty(self):
        return self.match_all(lambda line: Line(line).match_all(lambda value: value == Field.EMPTY))

def solve(board: Board):
    while not board.is_solved():
        for i in range(board.height):
            if board.is_solved():
                break
            chain_solve(board, i, True)
        for j in range(board.width):
            if board.is_solved():
                break
            chain_solve(board, j, False)
        input("\033[31mHasn't solved with chaining, enter to continue\033[0m")

def chain_solve(board: Board, index: int, row: bool):
    KEY = f'{"row" if row else "col"}{index}'
    print(KEY)
    line_getter: Callable[[int], GameLine] = board.get_row if row else board.get_column
    line_solver: Callable[[int], set[int]] = board.solve_row if row else board.solve_column
    line = line_getter(index)
    print(f"line = {line}")
    solved, diffs = line.is_solved_diffs()
    if solved:
        print("SOLVED")
    else:
        diffs = line_solver(index)
    print(f'diffs = {diffs}')
    board.latest_changes = {(diff if row else index, diff if not row else index) for diff in diffs}
    print(f'{board.latest_changes=}')
    board.print()
    if len(diffs) != 0: input()
    for i in diffs:
        print("chained")
        chain_solve(board, i, not row)
    if not board.is_solved():
        try:
            chain_solve(board,
                        ((index + 1) % (board.width if row else board.height)),
                        not row if index + 1 >= board.width else row)
        except Exception as e:
            board.print()
            print(e)


if __name__ == "__main__":

    with open("config.json", "rt") as f:
        data = json.loads(f.read())

    b = Board(**data)

    solve(b)

    print(*ProcedureLib.get_names(), sep="\n")
