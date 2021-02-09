p_left = '('
p_right = ')'
blk_l_delim = '['
blk_r_delim = ']'
comment = '//'
class AutoTree:
    def __init__(self, exp):
        ops = {p_left, p_right, blk_l_delim, blk_r_delim}
        whitespace = {' ', '\t', '\n'}
        def ignore_comments(exp, i):
            if (i < len(exp) - 1) and exp[i:i+2] == comment:
                while(i < len(exp) and exp[i] != '\n'):
                    i += 1
            return i

        def init(exp, i):
            ret = []
            while i < len(exp):
                if exp[i] == blk_l_delim:
                    tmp, i = init(exp, i+1)
                    ret.append(tmp)
                elif exp[i] == blk_r_delim:
                    return ret, i
                elif exp[i] == p_left:
                    tmp, i = init(exp, i+1)
                    ret.append(tmp)
                elif exp[i] == p_right:
                    return tuple(ret), i
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
        self.Patterns = dict()
        for p in tree:
            self.Patterns[p[0]] = Pattern(p)

    def get_root(self):
        pass

    def is_elusive(self, root_pattern):
        pass

    def is_satisfying(self, root_pattern):
        pass

class Pattern:
    def __init__(self, plst):
        self.plst = plst
        self.name = plst[0]
        self.debug = plst[1]
        self.unoccupied = plst[2]
        self.occupied = plst[3]
        self.tree = plst[4]

print(AutoTree("( pattern8 ((c6 BR) (d4 BR))(d6 e3 e4 e5 e6 f2 f3 f4 f5 f6 g1 g2 g3 g4 g5 g6)(c6 d4 BR)[(f3 [(pattern2ab (e3 e4) (d4 f3))][(pattern2ab (g2 g3) (f3 BR))])(e5 [(d6) (e4)][(pattern13 (e6 f4 f5 f6 g3 g4 g5 g6) (e5 BR))])(f2 [(pattern2ab (g1 g2) (f2 BR))][(pattern9 (g5 g4 f5 f4 f3 e5 e4 e3) (BR f2 d4))])(e3 [(pattern17 (d6 e5 e6 f2 f3 f4 f5 g1 g2 g3 g4 g5) (c6 d4 e3 BR))]) ])").Patterns["pattern8"].tree)
