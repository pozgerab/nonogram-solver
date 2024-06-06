from typing import Callable, Literal, Iterable, Self
from portion import Interval, iterate, closed
from list import List, IntList, FieldValueList, FieldValue

type Procedure = Callable[[GameLine], tuple[GameLine, Procedure | None]]  # A procedure callable


SIZECOMPARATOR: Callable[[List[any]], int] = lambda l: l.size()


class Field:
    """Field "enum\""""
    COLOR: FieldValue = 1
    EMPTY: FieldValue = 0
    EXCLUDE: FieldValue = -1


class Line(FieldValueList):
    def __init__(self, __iterable: Iterable[FieldValue]):
        super().__init__(__iterable)

    def validate(func):
        def isvalid(self: Self, index: int):
            if type(index) != int: print("HOOOOW")
            if not (0 <= index < self.size()): return
            func(self, index)

        return isvalid

    def get(self, index: int):
        return self[index]

    def setField(self, index: int, value: FieldValue):
        self[index] = value

    @validate
    def color(self, index: int):
        """
        Colors the field (set its value to 1)
        """
        if self[index] != Field.EXCLUDE:
            self[index] = Field.COLOR
        else:
            print("Trying to color an impossible field")

    @validate
    def exclude(self, index: int):
        '''
        Excludes the field (set its value to -1)
        '''
        if self[index] != Field.COLOR:
            self[index] = Field.EXCLUDE
        else:
            print("Trying to exclude a colored field")

    def getIndexesOfFieldValueMultiple(self, fields: set[FieldValue]) -> IntList:
        """
        Gets the indexes of the given field values in the line. [-1,-1,0,1] + (0,1) -> [2,3]
        """
        return IntList(range(self.size())).filter(lambda index: self[index] in fields)

    def getIndexesOfFieldValue(self, field: FieldValue) -> IntList:
        """
        Gets the indexes of a single field value in the line. [-1,-1,0,1] + 0 -> [2]
        """
        return self.getIndexesOfFieldValueMultiple({field})

    def groupConnectedFieldMultiple(self, fields: set[FieldValue]):
        """
        Groups multiple field values if they are connected. [0,0,-1,1,0,0] + (0,1) -> [[0,1], [3,4,5]]
        """
        list_: List[IntList] = List()
        fieldIndexes = self.getIndexesOfFieldValueMultiple(fields)
        collector = List()
        for i, v in enumerate(fieldIndexes):
            if collector.size() == 0:
                collector.append(v)
                continue
            if fieldIndexes[i - 1] == v - 1:
                collector.append(v)
                continue
            list_.append(collector)
            collector = List()
            collector.append(v)
        list_.append(collector)
        return list_

    def groupConnectedField(self, field: FieldValue) -> List[IntList]:
        """
        Groups the field values if they are connected. [0,0,-1,1,0,0] + 0 -> [[0,1], [4,5]]
        """
        return self.groupConnectedFieldMultiple({field})

    def getStartedGroups(self):
        """ O-OX--O -> [0,1,2], [4,5,6]"""
        return self.groupConnectedFieldMultiple({Field.COLOR, Field.EMPTY})

    def getFinishedGroups(self):
        """Groups multiple field only if all of them is colored\n
        O-O-X-OO -> []\n
        O-XOO -> [[3,4]]
        """
        return self.getStartedGroups().filter(lambda group: group.matchAll(lambda f: f == Field.COLOR))

    def getStartedWithColor(self):
        return self.getStartedGroups().filter(lambda group: group.contains(Field.COLOR))

    def getStartedWithColorButNotFinished(self):
        return self.getStartedWithColor().filter(lambda group: not group.matchAll(lambda f: f == Field.COLOR))

    def getEmptyStartedGroups(self):
        return self.getStartedGroups().filter(lambda group: not group.contains(Field.COLOR))

    def getFirstColoredIndexIfNotEmptyBefore(self):
        return self.getStartedGroups()[0][0] if self.get(self.getStartedGroups()[0][0]) == Field.COLOR else None

    def getLastColoredIndexIfNotEmptyAfterFromBackwards(self):
        sg = self.getStartedGroups()
        return self.size() - 1 - sg[-1][-1] if self.get(sg[-1][-1]) == Field.COLOR else None


class OffsetLine(Line):
    """
    Mergable line with an offset
    """

    def __init__(self, __iterable: Iterable[FieldValue], offset=0):
        super().__init__(__iterable)
        self.offset = offset

    def merge(self, line: Line) -> Line:
        """
        Merges this line to another line by its offset. Returns a new line instance. [-1,-1] + offset = 3 + [1,1,-1,0,0] -> [1, 1,-1,-1,-1]
        """
        newL = Line(line)
        for i in range(self.size()):
            newL[i + self.offset] = self[i]
        return newL


class GameLine(tuple[Line, IntList]):
    """
    Line with tasks. Solvable
    """

    def __init__(self, gameLine: tuple[Line, IntList]):
        self.line, self.task = gameLine
        self.task = IntList(self.task)

    def task_reducable(func: Procedure):

        def reduceTask(gLine: Self):

            originalTask = IntList(gLine.task)

            started = gLine.line.getStartedGroups()
            doneGroups = started.filter(lambda g: g.matchAll(lambda i: gLine.line[i] == Field.COLOR))
            for g in doneGroups:
                if g.size() in gLine.task:
                    gLine.task.remove(g.size())

            ret = func(gLine)

            gLine.task = originalTask

            return ret

        return reduceTask

    # # TODO: !!!!!!!!!!!! HA AZ ELSŐ getStarted()[0] ban == 1 akkor ugyanaz mintha az elején lenne a színes. Ez igaz
    #  hátrafele is

    def procIfDone(self):
        print(162)
        colors = self.line.count(Field.COLOR)
        colorIndex = self.line.getStartedGroups()
        if colors == self.task.addUp():
            for i in colorIndex.plainer():
                self.line.exclude(i)
        return self, None

    def procZero(self):  # DOCED
        print("169")
        coloredGroups = self.line.groupConnectedField(Field.COLOR)
        toCheck = (coloredGroups, self.line.getStartedGroups(), self.line.getStartedWithColor())
        for groups in toCheck:
            if groups.size() == self.task.size() and groups.matchAllWithIndex(
                    lambda index, intlist, _: intlist.size() == self.task[index]):
                indexes = groups.plainer()
                for i in range(self.line.size()):
                    self.line[i] = (Field.COLOR if indexes.contains(i) else Field.EXCLUDE)
                return self, None
        return self, None

    def procedureOne(self):  #   DOCED
        """
        Procedure one. Main starting procedure
        """
        print("Proc 1:177")

        if self.task.size() != 0:

            coloredGroups = self.line.getFinishedGroups()  # Színes mezők indexje
            connectedFieldArray: List[IntList] = self.line.groupConnectedField(Field.EMPTY)  # Üres merzők group
            emptyGroups = self.line.getEmptyStartedGroups()
            allConnected = emptyGroups.size() == 1 and emptyGroups.plainer().size() == connectedFieldArray.size()  # Ha összes össze van e kötve és az első eleme
            allBlock = emptyGroups[0] if allConnected else None
            workLine = OffsetLine(self.line, 0)  # Ezen a lineon operálunk
            workTasks = IntList(self.task)  # Ezeken a taskokon operálunk
            if allConnected:
                for i in coloredGroups:
                    if i.size() == 0: continue
                    workTasks.remove(i.size())
                workLine = OffsetLine([Field.EMPTY for _ in range(allBlock.size())],
                                      allBlock[0])  # Kivágja a -1 részeket

            taskSize = workTasks.size()  # Hány task van
            iterator = 0  # Iterator
            taskMap: dict[int, dict[Literal["start", "end"], int]] = {i: {"start": 0, "end": 0} for i in range(
                taskSize)}  # Minden taskhoz egy kezdő és végződő index kell majd, egyenlőre 0
            intervalMap: dict[int, Interval] = {}  # Minden taskhoz intervallum
            for i, t in enumerate(workTasks):  # Bejelöli a végeket
                taskMap[i]["end"] = iterator + t - 1
                iterator += t + 1
            iterator = workLine.size() - 1
            for i, t in enumerate(workTasks):  # Bejelöli a kezdőket
                taskMap[taskSize - i - 1]["start"] = iterator - workTasks[-i - 1] + 1
                iterator -= workTasks[-i - 1] + 1
            for i, v in taskMap.items():  # Végigmeg a taskMapen és intervallumba foglalja őket
                if v["start"] > workLine.size() - 1 or v["end"] > workLine.size() - 1: continue
                intervalMap[i] = closed(v["start"], v["end"])

            for p in intervalMap.values():  # Végigmegy az intervallumokon és kiszínezi a mezőket
                for field in iterate(p, 1):
                    workLine.color(field)
            self.line = workLine.merge(self.line)  # Mergeli a két line-t
        return self, None

    def procOneDotZero(self):  # DOCED
        print("procOneDotZero:214")

        def repeatFromStart():
            """
            Ha kisebb az első üres hely mint az első task -> Kizárja\n
            Ismétlődik amíg lehet
            """
            before = Line(self.line)
            emptyIndexes = self.line.getStartedGroups()
            if emptyIndexes.size() > 0 and 0 < emptyIndexes[0].size() < self.task[0]: [
                self.line.exclude(i) for i in emptyIndexes[
                    0]]  # Ha az első task nagyobb mint az első üres hely Végigmegy a kisebb üres helyek indexein és kizárja
            after = self.line
            if after == before:
                return True
            return False

        def repeatFromEnd():
            '''
            Ha kisebb az utolsó üres hely mint az utolsó task -> Kizárja\n
            Ismétlődik amíg lehet
            '''
            before = Line(self.line)
            emptyIndexes = self.line.getStartedGroups()
            if emptyIndexes.size() > 0 and 0 < emptyIndexes[-1].size() < self.task[-1]: [
                self.line.exclude(i) for i in emptyIndexes[
                    -1]]  # Ha az első task nagyobb mint az első üres hely  Végigmegy a kisebb üres helyek indexein és kizárja
            after = self.line
            if after == before:
                return True
            return False

        while not repeatFromStart(): break  # Ismétli
        while not repeatFromEnd(): break

        startedGs = self.line.getStartedGroups()

        smallerThanAnyTaskGs = startedGs.filter(lambda g: self.task.matchAll(lambda t: g.size() < t))
        for i in smallerThanAnyTaskGs:
            i.forEach(lambda v: self.line.exclude(v))

        return self, None

    def procIfSide(self):  # DOCED
        print("ifside:253")
        i, j = 0, 0
        firstCIndex = self.line.getFirstColoredIndexIfNotEmptyBefore()
        if firstCIndex is not None:
            for i in range(firstCIndex, firstCIndex + self.task[0]):
                self.line.color(i)
            self.line.exclude(i + 1)
        lastCIndex = self.line.getLastColoredIndexIfNotEmptyAfterFromBackwards()
        if lastCIndex is not None:
            for j in range(lastCIndex + 1, lastCIndex + self.task[-1]):
                self.line.color(-j)
            self.line.exclude(-j - 1)

        return self, None

    @task_reducable
    def procIfSameGAsTaskAndTCor(self):  # DOCED
        print(f'procIfSameGAsTaskAndTCor:269')
        started = self.line.getStartedWithColorButNotFinished()
        if started.size() == self.task.size():
            for i in range(self.task.size()):
                if started[i].countCondition(lambda f: f == Field.COLOR) == self.task[i]:
                    for f in range(started[i][0], started[i][-1] + 1):
                        if self.line[f] != Field.COLOR:
                            self.line.exclude(f)

    def procIfImpossibleG(self):  # DOCED
        print("procIfImpossibleG:279")
        if self.task.size() < self.line.getStartedGroups().size():  # Ha kevesebb task mint group
            notEmptyGroups = self.line.getStartedWithColor()  # Groupok amiben van színes
            onlyEmptyGroups = self.line.getEmptyStartedGroups()  # Groupok amiben nincs színes
            if notEmptyGroups.size() == self.task.size():
                [[self.line.exclude(i) for i in group] for group in
                 onlyEmptyGroups]  # Ha annyi színes group mint task, a többi group biztos nem jó

        return self, None

    def procIfOneIfBiggerThanAny(self):
        print("ifbiggerThanAny:288")
        started = self.line.getStartedGroups()
        biggestTask = self.task.max()
        bigGs = started.filter(lambda g: g.size() >= biggestTask)
        if bigGs.size() == 1:
            biggestG = bigGs[0]

            subLine = Line([self.line[i] for i in biggestG])
            subGameLine = GameLine((subLine, IntList.of(biggestTask)))
            subGameLine, _ = subGameLine.procedureOne()

            [self.line.setField(biggestG[i], v) for i, v in enumerate(subGameLine.line)]

        return self, None

    def procIfCertainG(self):
        print("procIfCertainG:305")
        startedGroups = self.line.getStartedGroups()
        notFinishedGroups = startedGroups.filter(
            lambda g: g.matchAny(lambda i: self.line[i] == Field.COLOR) and not g.matchAll(
                lambda i: self.line[i] == Field.COLOR))  # Nem teljesen színes groupok
        if notFinishedGroups.size() != 0 and notFinishedGroups[0].size() != 0:
            if self.task[0] == notFinishedGroups[0].size():  # Ha jó akkor színez
                [self.line.color(i) for i in notFinishedGroups[0]]

        return self, None

    procOneFour: Procedure = procIfCertainG

    def procedureTwo(self):  #   KIZÁR
        """
            KIZÁR
            ha egy task van kizárja a biztos nemeket
        """
        print("Proc 2")

        originalTask = IntList(self.task)
        colorGroup = self.line.groupConnectedField(Field.COLOR)
        if not colorGroup[0].isEmpty():
            colorSorroundedByEx = colorGroup.filterWithIndex(
                lambda i, g, a: (g[0] == 0 and (a[g[-1] + 1] if g[-1] + 1 < a.size() else 10) == Field.EXCLUDE) or (
                        g[-1] == a.size() - 1 and (a[g[0] - 1] if g[0] - 1 >= 0 else 10) == Field.EXCLUDE) or (
                                        (a[g[-1] + 1] if g[-1] + 1 < a.size() else 10) == Field.EXCLUDE and (
                                    a[g[0] - 1] if g[0] - 1 >= 0 else 10) == Field.EXCLUDE))

            originalTask = IntList(self.task)
            for g in colorSorroundedByEx:
                if len(g) in self.task:
                    self.task.remove(len(g))

        if self.task.size() != 1:
            self.task = originalTask
            return self, None  # Csak ha egy task van
        task = self.task[0]  # Egyetlen task szám
        coloredIndexes = self.line.getIndexesOfFieldValue(Field.COLOR)  # Színes mezők indexje
        if coloredIndexes.size() == task:  # Ha annyi színes amennyi kell -> minden más nem jo
            for i in range(self.line.size()):
                if not coloredIndexes.contains(i): self.line.exclude(i)

        if coloredIndexes.size() == 0:
            self.task = originalTask
            return self, None
        highestColored = coloredIndexes[-1]  # Legnagyobb index ahol színes
        lowestColored = coloredIndexes[0]  # Legkisebb index ahol színes
        if highestColored - lowestColored > 1:  # Ha nem egymás mellett van a két színezett
            if highestColored - lowestColored + 1 > task: raise Exception("Two colors cant be connected: Procedure 2")
            for i in range(lowestColored, highestColored):
                self.line.color(i)  # Összeköti a két színeset
        if highestColored + 1 > task:
            [self.line.exclude(i) for i in range(highestColored - task + 1)]  # Ha nem éri el a szélét a legnagyobb
            # indextol, tehát van mit kizárni # Kizárja a kizárandó mezőket
        if self.line.size() - 1 - lowestColored >= task: [self.line.exclude(i) for i in
                                                          range(self.line.size() - 1 - lowestColored + task - 1,
                                                                self.line.size())]  # Ha nem éri el a legkisebb
        # indextol a nagyobb szélét, tehát van mit kizárni # Kizárja a (lehetséges után + 1) tól a végéig(size())

        self.task = originalTask
        return self, None


class Board(List[Line]):
    '''Játéktábla'''

    def __init__(self, width=10, height=10):
        super().__init__()
        self.height = height
        self.width = width
        self.rowTask: List[IntList] = [[1, 3], [1, 4], [1, 3, 2], [3, 3, 2], [3, 1], [5], [3], [1, 3], [2, 3],
                                       [6]]  # HARD CODED
        self.colTask: List[IntList] = [[4, 2], [1, 3], [3, 1], [1, 6], [8], [2, 5], [1, 1, 1], [2], [5],
                                       [4]]  # HARD CODED
        [self.append([0 if j == 3 else 0 for j in range(width)]) for i in range(height)]

    def content(self):
        """
        Listaként a mezők
        """
        return List(self)

    @DeprecationWarning
    def clone(self):
        """
        Klónozza. Ez nem működik
        """
        board = Board(self.width, self.height)
        board.rowTask = self.rowTask
        board.colTask = self.colTask
        return board

    def __repr__(self) -> str:
        return f'Board[][]::\n{[i for i in self]}'

    def print(self):
        """Vizuálisan megjeleníti a játéktáblát"""
        print("Board[][]::\n")
        for i in range(self.height):
            print("".join(["▁" if self[i][j] == 0 else "█" if self[i][j] == 1 else "◍" for j in range(self.width)]),
                  sep="")

    def getColumn(self, index: int) -> GameLine:
        """Get column by index"""
        return GameLine((Line(self[index]), self.colTask[index]))

    def getRow(self, index: int) -> GameLine:
        """Get row by index"""
        return GameLine((Line([self[c][index] for c in range(self.width)]), self.rowTask[index]))

    def setColumn(self, index: int, line: Line):
        """Set the column at index"""
        self[index] = line

    def setRow(self, index: int, line: Line):
        """Set the row at index"""
        for c in range(self.width):
            self[c][index] = line[c]

    def getField(self, column=0, row=0):
        """Get the specific field by x and y coordinates"""
        return self[column][row]

    @staticmethod
    def procedures() -> list[Procedure]:
        """Register the procedures here. For now, it runs in strict order"""
        return [GameLine.procIfDone, GameLine.procedureOne, GameLine.procZero, GameLine.procOneDotZero,
                GameLine.procIfSide, GameLine.procIfImpossibleG, GameLine.procOneFour, GameLine.procIfOneIfBiggerThanAny
                #GameLine.procedureTwo
                ]

    def solveRow(self, index: int):
        '''Applies the procedures to a row at the index and replaces it with the previous one. Return if the line changed'''
        before = self.getRow(index)
        for proc in Board.procedures():
            self.setRow(index, proc(self.getRow(index))[0].line)
        return self.getRow(index) != before

    def rowProc(self, index: int, proc: Procedure):
        """Applies the given procedure to a row at a given index. Return if the line changed"""
        before = self.getRow(index)
        self.setRow(index, proc(self.getRow(index))[0].line)
        return self.getRow(index) != before

    def colProc(self, index: int, proc: Procedure):
        """Applies the given procedure to a column at a given index. Return if the line changed"""
        before = self.getColumn(index)
        self.setColumn(index, proc(self.getColumn(index))[0].line)
        return self.getColumn(index) != before

    def solveColumn(self, index: int):
        """Applies the procedures to a column at the index and replaces it with the previous one. Return if the line
        changed"""
        before = self.getColumn(index)
        for proc in self.procedures():
            self.setColumn(index, proc(self.getColumn(index))[0].line)
        return self.getColumn(index) != before

    def isSolved(self):
        """Checks if any field is empty, doesn't actually check if the field are correct"""
        return self.matchAll(lambda line: not line.contains(0))

    def isEmpty(self):
        return self.matchAll(lambda line: Line(line).matchAll(lambda value: value == Field.EMPTY))


b = Board(10, 10)


def chainSolve(board: Board, index: int, row: bool, boardCheck: dict):
    KEY = f'{"row" if row else "col"}{index}'
    print(KEY)
    if boardCheck.get(KEY) == (prev := (board.getRow if row else board.getColumn)(index).line):
        return False
    for proc in Board.procedures():
        (board.rowProc if row else board.colProc)(index, proc)
    line, _ = (board.getRow if row else board.getColumn)(index)
    diff = line.getDiffIndexes(prev)
    boardCheck.update({KEY: line})
    board.print()
    diff.forEach(lambda lindex: chainSolve(board, lindex, not row, boardCheck))
    if board.isEmpty():
        return chainSolve(board, index + 1 if (switchDir := index < board.width) else 0, not row if switchDir else row,
                          boardCheck)
    return True


chainSolve(b, 0, True, {})
b.print()
