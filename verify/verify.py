from copy import copy

from board import Position, BCH


conj = "and"
disj = "or"
class Node:
    def __init__(self, expr):
        self.op = expr[0]
        lb = 2
        if self.op == conj:
            self.move = expr[1]
            self.cells = {self.move}
        elif self.op == disj:
            self.move = None
            self.cells = set()
            lb = 1
        else:
            raise Exception("Invalid operation.")
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
b_left = '['
b_right = ']'
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


    def expand(self, root):
        r = copy(root)
        if not r.children:
            return r
        ch = Node([disj])
        r.children = [ch]

        gch = []
        for i in range(len(root.children)):
            cch = root.children[i].children
            for c in cch:
                newc = copy(c)
                newc.children = copy(c.children)
                newc.children.extend(root.children[:i] + root.children[i+1:])
                gch.append(newc)

        ch.children = gch
        for i in range(len(ch.children)):
            ch.children[i] = self.expand(ch.children[i])

        union = {root.move}
        for ch in root.children:
            union.union(ch.cells)
        r.cells = union
        return r


    def _is_satisfying(self, board, root):
        if not board.requestmove(BCH + ' ' + root.move):
            return False
        #board.showboard()
        if not root.children:
            w = board.has_win(BCH)
            if not w:
                print("Not satisfying:")
                board.showboard()
            board.undo()
            return w

        for i in range(len(root.children)):
            cch = root.children[i].children
            for c in cch:
                newc = copy(c)
                newc.children = copy(c.children)
                newc.children.extend(root.children[:i] + root.children[i+1:])
                if not self._is_satisfying(board, newc):
                    board.undo()
                    return False

        board.undo()
        return True


    def is_satisfying(self):
        return self._is_satisfying(Position(self.rows, self.cols), self.tree)


    def _is_elusive(self, root):
        for ch in root.children:
            if not self._is_elusive(ch):
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


    def is_elusive(self):
        return self._is_elusive(self.tree)


    def verify(self):
        # Check elusiveness first
        return self.is_elusive() and self.is_satisfying()




bs_v = "boardsize"
class Parser:
    def __init__(self, s, r_pn):
        self.var = {bs_v: None}
        self.root_pattern = r_pn
        self.ops = {p_left, p_right, b_left, b_right}
        self.whitespace = {' ', '\t', '\n'}
        self.tok, _ = self._tok(s, 0)
        self.patterns = {}
        self.translated = None


    def ignore_comments(self, exp, i):
        if (i < len(exp) - 1) and exp[i:i+2] == comment:
            while(i < len(exp) and exp[i] != '\n'):
                i += 1
        return i


    #TODO: matching brackets
    def _tok(self, exp, i):
        ret = []
        while i < len(exp):
            if exp[i] == p_left or exp[i] == b_left:
                tmp, i = self._tok(exp, i+1)
                ret.append(tmp)
            elif exp[i] == p_right:
                return tuple(ret), i
            elif exp[i] == b_right:
                return ret, i
            elif exp[i] in self.whitespace:
                pass
            else:
                tmp = ''
                while i < len(exp):
                    i = self.ignore_comments(exp, i)
                    if exp[i] in self.ops or exp[i] in self.whitespace:
                        i -= 1
                        break
                    tmp += exp[i]
                    i += 1
                if tmp:
                    ret.append(tmp)
            i += 1
        return (ret, i)


    def is_pt(self, l):
        #TODO regular expression
        return len(l) >= 2 and l[0].isalpha() and l[1:].isnumeric()


    #TODO: disallow non-position elements in occupied/mappings
    def parse(self):
        for command in self.tok:
            if len(command) == 2 and type(command[0]) == str and command[0] in self.var:
                self.var[command[0]] = command[1]
            elif len(command) != 5 or type(command[0]) != str:
                print(command)
                raise Exception("Bad Pattern.")

            self.patterns[command[0]] = command

        if self.root_pattern not in self.patterns:
            raise Exception("No root pattern?")

        ptn = self.patterns[self.root_pattern]
        if type(ptn) != tuple or len(ptn) < 3 or type(ptn[2]) != tuple or [x for x in ptn[2] if type(x) != str]:
            print(ptn)
            raise Exception("Bad root pattern")

        tree = ([i for i in ptn[3] if self.is_pt(i)][0], self.parse_pattern(ptn, {x:x for x in ptn[2]}))
        self.translated = self.translate(tree)


    def translate(self, l):
        if type(l) == str:
            return l + " "
        ret = " "
        for i in l:
            ret += self.translate(i)
        mp = {list:disj, tuple:conj}
        return p_left + mp[type(l)] + ret + p_right


    # TODO: disallow circular patterns
    def parse_pattern(self, ptn, mapping):
        name = ptn[0]
        if type(name) != str:
            print(ptn)
            raise Exception("Bad pattern name.")
        debug = ptn[1]
        unoccupied = ptn[2]
        if type(unoccupied) != tuple or [x for x in unoccupied if type(x) != str]:
            print(ptn)
            raise Exception("Bad unoccupied cell list.")
        occupied = ptn[3]
        if type(occupied) != tuple or [x for x in occupied if type(x) != str]:
            print(ptn)
            raise Exception("Bad occupied list.")
        tr = ptn[4]
        if type(tr) != list:
            print(ptn)
            raise Exception("Invalid tree")

        return self.parse_expr(tr, mapping)


    def parse_expr(self, expr, mapping):
        tmp = []
        if type(expr) == str:
            if expr in mapping:
                return mapping[expr]
            return expr
        elif type(expr) == tuple:
            tmp = ()
        for e in expr:
            psed = self.parse_expr(e, mapping)
            if type(tmp) == list and type(psed) == list:
                return psed
            if type(e) in {tuple, list}:
                if len(expr) == 3 and type(expr[0]) == str and expr[0] in self.patterns and type(expr[1]) == tuple and len(expr[1]):
                    ptn_m = self.patterns[expr[0]][2]
                    mp = {}
                    for i in range(len(ptn_m)):
                        mp[ptn_m[i]] = mapping[expr[1][i]]
                    return self.parse_pattern(self.patterns[expr[0]], mp)

            t = type(tmp)
            tmp = t(list(tmp) + [psed])
        return tmp
    

#4x4
#tr = "(and c2 (or c1 d1) (or (and b3 (or a4 b4)) (and d3 (or d2 c3) (or c4 d4))))"
#tr = "(and d1 (or (and c3 (or d2 c2) (or b4 c4)) (and b3 (or a4 b4) (or c2 (and b2 (or b1 c1)))) (and d2 (or (and d3 (or c4 d4)) (and b3 (or a4 b4) (or c3 (and b2 (or b1 c1)))))) (and d3 (or c4 d4) (or d2 (and b3 (or a4 c3) (or c2 (and b2 (or b1 c1))))))))"

#3x3
#tr = "(and b2 (or (and b1 (or a3 b3)) (and c1 (or a3 b3)) (and a3 (or b1 c1)) (and b3 (or b1 c1))))"
#tr = "(and b2 (or b1 c1) (or a3 b3))"
#tr = "(and a2 (or a1 b1) (or a3 (and c2 (or b2 c1) (or b3 c3))))"

#autotr = AndOrAutoTree(tr, 4, 4)
#print(autotr)
#print()
#print(autotr.expand(autotr.tree))
#print()
#print(autotr.verify())

p = Parser('''
(boardsize 4)
( 
  vc 
  ((a1 b2))
  (a2 b1)
  (a1 b2)

  [(a2) (b1)]
)

(
  4x4proof
  ((BL TR))
  (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d2 d3 d4)  
  (BL TR d1)

  [
     (c3 [ (vc(d2 c2)(d1 c3)) ]
         [ (vc(b4 c4)(c3 TR)) ])
     
     (b3 [(a4) (b4)]
         [(c2) (b2 [(b1) (c1)])])

     (d2 [(d3 [(c4) (d4)])
          (b3 [(a4) (b4)]
              [(c3) (b2 [(b1) (c1)])]
          )
         ])    

     (d3 [(c4) (d4)]
         [(d2) 
	  (b3 [(a4) (c3)]
              [(c2) 
               (b2 [(b1) (c1)]) 
              ])
         ])
   ]
)
''', "4x4proof")
p.parse()
print(p.translated)
size = int(p.var[bs_v])
autotr = AndOrAutoTree(p.translated, size, size)
print(autotr.verify())
