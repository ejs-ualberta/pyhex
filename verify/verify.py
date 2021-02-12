conj = "and"
disj = "or"

from board import Position, BCH

class Node:
    def __init__(self, expr):
        self.op = expr[0]
        lb = 2
        if self.op == conj:
            self.move = expr[1]
            self.cells = {self.move}
        else:
            self.move = None
            self.cells = set()
            lb = 1
        self.children = []
        for item in expr[lb:]:
            if type(item) == str:
                self.cells.add(item)
                self.children.append(Node([conj, item]))
            elif type(item) == Node:
                self.cells = self.cells.union(item.cells)
                self.children.append(item)
        for item in self.children:
            if item.op == self.op:
                raise Exception("Node types do not alternate.")
        #print(self.cells)

    def __str__(self):
        contents = ""
        if self.op == conj:
            contents = ' ' + self.move
        for ch in self.children:
            contents += ' ' + str(ch)
        return '(' + self.op + contents + ')'

p_left = '('
p_right = ')'
comment = '//'
class AndOrAutoTree:
    def __str__(self):
        return str(self.tree)


    def __init__(self, exp, rows, cols):
        self.rows = rows
        self.cols = cols
        ops = {p_left, p_right}
        whitespace = {' ', '\t', '\n'}
        def ignore_comments(exp, i):
            if (i < len(exp) - 1) and exp[i:i+2] == comment:
                while(i < len(exp) and exp[i] != '\n'):
                    i += 1
            return i

        def init(exp, i):
            ret = []
            while i < len(exp):
                if exp[i] == p_left:
                    tmp, i = init(exp, i+1)
                    ret.append(tmp)
                elif exp[i] == p_right:
                    return Node(ret), i
                elif exp[i] in whitespace:
                    pass
                else:
                    tmp = ''
                    while i < len(exp):
                        i = ignore_comments(exp, i)
                        if exp[i] in ops or exp[i] in whitespace:
                            i -= 1
                            break
                        tmp += exp[i]
                        i += 1
                    if tmp:
                        ret.append(tmp)
                i += 1
            return (ret, i)
        tree, _ = init(exp, 0)
        self.tree = tree[0]


    def _is_satisfying(self, board, nd):
        if nd.op == conj:
            board.requestmove(BCH + ' ' + nd.move)
        if not nd.children:
            w = board.has_win(BCH)
            board.undo()
            return w
        for ch in nd.children:
            s = self._is_satisfying(board, ch)
            if not s:
                return False
        if nd.op == conj:
            board.undo()
        return True


    def is_satisfying(self):
        #root = self.expand()
        root = self.tree
        board = Position(self.rows, self.cols)
        return self._is_satisfying(board, root)


    def is_elusive(self, root=None):
        if not root:
            root = self.tree
        for ch in root.children:
            if not self.is_elusive(ch):
                return False

        if root.op == conj:
            for i in range(len(root.children)):
                for j in range(i+1, len(root.children)):
                    ch1 = root.children[i]
                    ch2 = root.children[j]
                    if ch1.cells.intersection(ch2.cells):
                        print("Not elusive:\n--> ", end='')
                        print(root)
                        return False
                           
        elif root.op == disj:
            # or nodes must have at least two children, otherwise
            # there is no point using an or node.
            if len(root.children) < 2:
                print("Or node has less than 2 children\n--> ", end='')
                print(root)
                return False
            isect = root.children[0].cells
            for ch in root.children[1:]:
                isect = isect.intersection(ch.cells)
            if isect:
                print("Not elusive:\n--> ", end='')
                print(root)
                return False
        return True
            

tr = "(and b2 (or (and b1 (or a3 b3)) (and c1 (or a3 b3)) (and a3 (or b1 c1)) (and b3 (or b1 c1))))"
#tr = "(and b2 (or b1 c1) (or a3 b3))"
autotr = AndOrAutoTree(tr, 3, 3)
print(autotr.is_satisfying())
