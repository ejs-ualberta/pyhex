conj = "and"
disj = "or"

class Node:
    def __init__(self, expr):
        self.op = expr[0]
        lb = 2
        if self.op == conj:
            self.move = expr[1]
        else:
            self.move = None
            lb = 1
        self.children = []
        self.cells = {self.move}
        for item in expr[lb:]:
            if type(item) == str:
                self.cells.add(item)
                self.children.append(Node([conj, item]))
            elif type(item) == Node:
                self.cells = self.cells.union(item.cells)
                self.children.append(item)
        #print(self.cells)

p_left = '('
p_right = ')'
comment = '//'
class AndOrAutoTree:
    def __init__(self, exp):
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

    def is_elusive(self, root_pattern):
        pass

    def is_satisfying(self, root_pattern):
        pass

print(AndOrAutoTree("(and b2 (or b1 c1) (or a3 b3))").tree.children[1].children[0].move)
