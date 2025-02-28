import json
from collections.abc import Callable, Iterable
from typing import Literal, Self, Tuple

from portion import Interval, iterate, closed

from list import List, IntList, FieldValue

type ProcReturn = tuple[GameLine, set[int]]
type ProcInput = [GameLine]
type Solverithm = Callable[ProcInput, GameLine]
type Procedure = Callable[ProcInput, ProcReturn]  # A procedure callable


class Field:
    """Field "enum\""""
    COLOR: FieldValue = 1
    EMPTY: FieldValue = 0
    EXCLUDE: FieldValue = -1

    @staticmethod
    def values():
        return {Field.COLOR, Field.EMPTY, Field.EXCLUDE}


class Line(List[Literal[-1, 0, 1]]):
    def __init__(self, __iterable: Iterable[FieldValue]):
        super().__init__(__iterable)

    def __and__(self, other: Self):
        if isinstance(other, type(self)) and self.size() == other.size():
            return Line([self[i] or other[i] for i in range(self.size())])

    def get_index_if_value_is(self, index: int, value: FieldValue):
        return index if self[index] == value else None

    @staticmethod
    def indexes(func):
        """Indicates that the returned value are indexes (0 <= x < size)"""
        return func

    @staticmethod
    def values(func):
        """Indicates that the returned values are the field values (-1 <= x <= 1)"""
        return func

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

    def in_bound_lower(self, index: int):
        return 0 <= index

    def in_bound_upper(self, index: int):
        return index < self.size()

    def in_bound(self, index: int):
        return 0 <= index < self.size()

    def get(self, index: int):
        return self[index]

    def set_field(self, index: int, value: FieldValue):
        self[index] = value

    def values_from_indexes(self, indexes: IntList) -> Self:
        return Line([self[i] for i in indexes])

    @validate
    def fill(self, index: int):
        """
        Colors the field (set its value to 1)
        """
        if self[index] != Field.EXCLUDE:
            self[index] = Field.COLOR
        else:
            print("Trying to color an impossible field")

    @validate
    def exclude(self, index: int):
        """
        Excludes the field (set its value to -1)
        """
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
        list_: List[IntList] = List()
        field_indexes = self.indexes_of_fields(fields)
        collector = IntList()
        for i, v in enumerate(field_indexes):
            if collector.size() == 0:
                collector.append(v)
                continue
            if field_indexes[i - 1] == v - 1:
                collector.append(v)
                continue
            list_.append(collector)
            collector = List()
            collector.append(v)
        list_.append(collector)
        return list_

    @indexes
    def group_adjacent_field(self, field: FieldValue) -> List[IntList]:
        """
        Groups the field values if they are connected. [0,0,-1,1,0,0] + 0 -> [[0,1], [4,5]]
        """
        return self.group_adjacent_fields({field})

    @indexes
    def slice_by(self, field: FieldValue):
        return self.group_adjacent_fields(Field.values() - {field})

    @indexes
    def slice_by_x(self):
        """ O-OX--O -> [0,1,2], [4,5,6]"""
        return self.slice_by(Field.EXCLUDE)

    @indexes
    def slice_by_x_contains_o(self):
        return self.slice_by_x().filter(lambda group: Field.COLOR in self.values_from_indexes(group))

    @indexes
    def slice_by_x_contains_o_not_all_o(self):
        return self.slice_by_x_contains_o().filter(lambda group: not self.values_from_indexes(group).match_all(lambda fv: fv == Field.COLOR))

    @indexes
    def slice_by_x_all_empty(self):
        return self.slice_by_x().filter(lambda group: Field.COLOR not in self.values_from_indexes(group))

    @indexes
    def first_color_index_after_only_x(self):
        return self.get_index_if_value_is(self.slice_by_x()[0][0], Field.COLOR)

    @indexes
    def last_o_index_before_only_x(self):
        return self.get_index_if_value_is(self.slice_by_x()[-1][-1], Field.COLOR)


class ProcedureLib:
    PROCEDURES: dict[str, Procedure] = dict()
    @staticmethod
    def get_names() -> set[str]:
        return set(ProcedureLib.PROCEDURES.keys())

class GameLine(tuple[Line, IntList]):
    """
    Line with tasks. Solvable
    """

    def __init__(self, line: Line, task: IntList):
        self.line, self.task = line, IntList(task)

    def __new__(cls, line: Line, task: IntList):
        instance = super().__new__(cls, (line, task))
        return instance

    def __repr__(self):
        return f'l:{self.line}, t:{self.task}'

    def clone(self):
        return self.__class__(Line(self.line), IntList(self.task))

    @staticmethod
    def procedure(name: str):
        def decorator(procedure: Solverithm):
            def wrapper(game_line) -> ProcReturn:
                before = Line(game_line.line)
                after: GameLine = procedure(game_line)
                diffs = before.get_diff_indexes(after.line)
                print(f"Procedure '{name}' executed. Diffs: {diffs}")
                return after, diffs

            ProcedureLib.PROCEDURES[name] = wrapper
            return wrapper

        return decorator


    def get_too_small_groups(self):
        return self.line.slice_by_x_all_empty().filter(lambda group: self.task.match_all(lambda task: task > group.size()))

    def is_solved(self):
        return self.task == self.line.group_adjacent_field(Field.COLOR).map(lambda gr, i: gr.size())

    @procedure(name = "exclude if solved")
    def procIfDone(self):
        if self.is_solved():
            for ind in self.line.indexes_of_field(Field.EMPTY):
                self.line.exclude(ind)
        return self

    @procedure("if empty group = task amount && group size = task -> color all")
    def procZero(self):
        groups = self.line.slice_by_x()
        if groups.size() == self.task.size() and groups.matchAllWithIndex(
                lambda index, group, _: group.size() == self.task[index]):
            indexes = groups.flat()
            for i in range(self.line.size()):
                if i in indexes:
                    self.line[i] = Field.COLOR
        return self

    @procedure("remove too small groups")
    def removeTooSmall(self):
        for small_g in self.get_too_small_groups():
            for i in small_g:
                self.line.exclude(i)
        return self

    @procedure("default starting interval solve")
    def procedureOne(self):

        taskSize = self.task.size()
        slicedbyx = self.line.slice_by_x()
        original: Line | None = None
        if slicedbyx.size() == 1:
            original = Line(self.line)
            self.line = Line([self.line[i] for i in slicedbyx[0]])

        iterator = 0
        taskMap = {i: {"start": 0, "end": 0} for i in range(
            taskSize)}  # Minden taskhoz egy kezdő és végződő index kell majd, egyenlőre 0
        intervalMap: dict[int, Interval] = {}  # Minden taskhoz intervallum
        for index, task in enumerate(self.task):  # Bejelöli a végeket
            taskMap[index]["end"] = iterator + task - 1
            iterator += task + 1
        iterator = self.line.last_o_index_before_only_x() or self.line.size() - 1
        for index, task in enumerate(self.task):  # Bejelöli a kezdőket
            taskMap[taskSize - index - 1]["start"] = iterator - self.task[-index - 1] + 1
            iterator -= self.task[-index - 1] + 1
        for index, v in taskMap.items():  # Végigmeg a taskMapen és intervallumba foglalja őket
            if v["start"] > self.line.size() - 1 or v["end"] > self.line.size() - 1: continue
            intervalMap[index] = closed(v["start"], v["end"])

        for p in intervalMap.values():  # Végigmegy az intervallumokon és kiszínezi a mezőket
            for field in iterate(p, 1):
                self.line.fill(field)

        if original:
            replace_start = slicedbyx[0][0] # First index that got modified in case of trimming
            for i in range(self.line.size()):
                original[replace_start + i] = self.line[i]
            self.line = original
        return self

    @procedure("if color on side -> its always the start")
    def procIfSide(self):
        i, j = 0, 0
        firstCIndex = self.line.first_color_index_after_only_x()
        if firstCIndex is not None:
            for i in range(firstCIndex, firstCIndex + self.task[0]):
                self.line.fill(i)
            self.line.set_in_bound(i + 1, Field.EXCLUDE)

        lastCIndex = self.line.last_o_index_before_only_x()
        if lastCIndex is not None:
            for j in range(lastCIndex - self.task[-1] + 1, lastCIndex):
                self.line.fill(j)
            self.line.set_in_bound(lastCIndex - self.task[-1], Field.EXCLUDE)

        return self

    @procedure("if near start or end then continue the sequence")
    def continuestartorend(self):
        if Field.COLOR not in self.line:
            return self
        slicex = self.line.slice_by_x()
        original: Line | None = None
        if slicex.size() == 1:
            slice_line = slicex[0]
            original = Line(self.line)
            self.line = Line([self.line[i] for i in slice_line])

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

        if original:
            for i, og_index in enumerate(slice_line):
                original[og_index] = self.line[i]
            self.line = original

        return self

    @procedure("if biggest task can be only in one group -> run default start proc")
    def procIfOneIfBiggerThanAny(self):
        started = self.line.slice_by_x()

        biggestTask = max(self.task)
        bigGs = started.filter(lambda g: g.size() >= biggestTask)
        if bigGs.size() == 1:
            biggestG = bigGs[0]

            subLine = Line([self.line[i] for i in biggestG])
            subGameLine = GameLine(subLine, IntList.of(biggestTask))
            subGameLine.procedureOne()

            [self.line.set_field(biggestG[i], v) for i, v in enumerate(subGameLine.line)]

        return self

    @procedure("if first or last group contains color and is the size of the first task -> color")
    def procIfCertainG(self):
        groups = self.line.slice_by_x()
        if Field.COLOR in self.line.values_from_indexes(groups[0]) and groups[0].size() == self.task[0]:
            for i in groups[0]:
                self.line.fill(i)
        if Field.COLOR in self.line.values_from_indexes(groups[-1]) and groups[-1].size() == self.task[-1]:
            for i in groups[-1]:
                self.line.fill(i)

        return self

    @procedure("if task == started group -> color certain, exclude impossible")
    def procgroupeqtask(self):
        started_groups = self.line.slice_by_x_contains_o()
        if started_groups.size() == self.task.size():
            # Exclude every group that is not started
            [self.line.exclude(field) for field in range(self.line.size()) if field not in started_groups.flat()]

            for group, task in zip(started_groups, self.task):
                print(group, task)
                if group.size() == task:
                    for field in group:
                        self.line.fill(field)
                elif self.line.values_from_indexes(group).count(Field.COLOR) == task:
                    [self.line.exclude(field) for field in group if self.line != Field.COLOR]
                else:
                    line = Line([self.line[i] for i in group])
                    small_line = GameLine(line, IntList([task]))
                    small_line.procedureOne()
                    for i, v in zip(group, small_line.line):
                        self.line[i] = v
        return self

    @procedure("if an adjacent colored list is same size as biggest task -> surround with x")
    def surroundIfBiggest(self):
        colorGroups = self.line.group_adjacent_field(Field.COLOR)
        biggest_task = max(self.task)
        for group in colorGroups:
            if group.size() == biggest_task:
                for edge in (group[0] - 1, group[-1] + 1): #    Two surrounding fields
                    if self.line.size() > edge >= 0 == self.line[edge]:
                        self.line.exclude(edge)

        return self


    @procedure("if one task and at least one colored field -> exclude unreachable")
    def excludeunreachable(self):
        if self.task.size() != 1 or Field.COLOR not in self.line:
            return self
        task = self.task[0] # only task

        colored_fields = self.line.indexes_of_field(Field.COLOR)
        first_colored, last_colored = colored_fields[0], colored_fields[-1]
        for i in range(first_colored + 1, last_colored): # connect fields
            self.line.fill(i)

        excluded_from_start = range(0, max(last_colored - task + 1, 0))
        excluded_from_end = range(min(first_colored + task, self.line.size()), self.line.size())
        excluded = set().union(excluded_from_end, excluded_from_start)

        for exc_iter in excluded:
            self.line.exclude(exc_iter)

        return self

    #   TODO:   HA ELSO VAGY UTOLSO LEGNAGYOBB TASK ÉS VAN OLYAN COLOR BLOCK AMI AZ ÖSSZESNÉL NAGYOBB -> excludeunreachable
    #   TODO:   HA A TOOSMALLTASK EGY NAGYOBB BLOCK ELŐTT


class Board(List[Line]):
    """Játéktábla"""

    def __init__(self, width=10, height=10, **kwargs):
        super().__init__()
        self.height = kwargs.get("height") or height
        self.width = kwargs.get("width") or width
        self.rowTask: List[IntList] = kwargs.get("rowTask")
        self.colTask: List[IntList] = kwargs.get("colTask")
        [self.append([0 for _ in range(width)]) for _ in range(height)]

    def get(self, x: int, y: int):
        return self[x][y]

    def __getitem__(self, index: int | tuple[Ellipsis, int]):
        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, tuple):
            return Line([super().__getitem__(c)[index[1]] for c in range(self.width)])

    def content(self):
        """
        listaként a mezők
        """
        return List(self)

    def __repr__(self) -> str:
        return f'Board[][]::\n{[i for i in self]}'

    def print(self):
        print("Board[][]::\n")
        for i in range(self.height):
            print("".join(["▁" if self[i][j] == 0 else "█" if self[i][j] == 1 else "░" for j in range(self.width)]),
                  sep="")

    def get_column(self, index: int) -> GameLine:
        """Get column by index"""
        return GameLine(self[..., index], self.rowTask[index])

    def get_row(self, index: int) -> GameLine:
        """Get row by index"""
        return GameLine(Line(self[index]), self.colTask[index])

    def set_column(self, index: int, line: Line):
        """Set the column at index"""
        for c in range(self.width):
            self[c][index] = line[c]

    def set_row(self, index: int, line: Line):
        """Set the row at index"""
        self[index] = line

    def get_field(self, column=0, row=0):
        """Get the specific field by x and y coordinates"""
        return self[column][row]

    @staticmethod
    def procedures() -> set[Procedure]:
        """Register the procedures here. For now, it runs in strict order"""
        return set(ProcedureLib.PROCEDURES.values())

    def solve_row(self, index: int) -> set[int]:
        """Applies the procedures to a row at the index and replaces it with the previous one. Return the changed indexes"""
        diffs = set[int]()
        for proc in Board.procedures():
            game_line, proc_diffs = proc(self.get_row(index))
            self.set_row(index, game_line.line)
            diffs = diffs.union(proc_diffs)
        return diffs

    def row_proc(self, index: int, proc: Procedure):
        """Applies the given procedure to a row at a given index. Returns the changed indexes"""
        game_line, diff = proc(self.get_row(index))
        self.set_row(index, game_line.line)
        return diff

    def col_proc(self, index: int, proc: Procedure):
        """Applies the given procedure to a column at a given index. Returns the changed indexes"""
        game_line, diff = proc(self.get_column(index))
        self.set_column(index, game_line.line)
        return diff

    def solve_column(self, index: int) -> set[int]:
        """Applies the procedures to a column at the index and replaces it with the previous one. Return if the line
        changed"""
        diffs = set[int]()
        for proc in self.procedures():
            game_line, proc_diffs = proc(self.get_column(index))
            self.set_column(index, game_line.line)
            diffs = diffs.union(proc_diffs)
        return diffs

    def is_solved(self):
        for i in range(self.height):
            if not self.get_row(i).is_solved():
                return False
        return True

    def is_empty(self):
        return self.match_all(lambda line: Line(line).match_all(lambda value: value == Field.EMPTY))

def solve(board: Board):
    #while not board.is_solved():
        for i in range(board.height):
            if board.is_solved():
                break
            chain_solve(board, i, True)
        for j in range(board.width):
            if board.is_solved():
                break
            chain_solve(board, j, False)

def chain_solve(board: Board, index: int, row: bool):
    KEY = f'{"row" if row else "col"}{index}'
    print(KEY)
    line_getter: Callable[[int], GameLine] = board.get_row if row else board.get_column
    line_setter: Callable[[int, Line], None] = board.set_row if row else board.set_column
    line_solver: Callable[[int], set[int]] = board.solve_row if row else board.solve_column
    print(line := line_getter(index))
    if line.is_solved():
        game_line, diffs = line.procIfDone()
        line_setter(index, game_line.line)
        print("SOLVED")
    else:
        diffs = line_solver(index)
    print(f'diffs = {diffs}')
    board.print()
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

    board.print()


if __name__ == "__main__":
    #print(Line([1,0,0,0,1,0]).get_dist())

    with open("config.json", "rt") as f:
        data = json.loads(f.read())

    b = Board(**data)

    solve(b)
    b.print()

    print(*ProcedureLib.get_names(), sep="\n")
    
    #gameline = GameLine(Line([0, 0, 0, 0,0,1,1,-1]), IntList([5]))
    #print(gameline.continuestartorend())
