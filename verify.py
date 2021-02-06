p_left = '('
p_right = ')'
blk_l_delim = '['
blk_r_delim = ']'
class AutoTree:
    def __init__(self, exp):
        ops = {p_left, p_right, blk_l_delim, blk_r_delim}
        whitespace = {' ', '\t', '\n'}
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
                        if exp[i] in ops or exp[i] in whitespace:
                            i -= 1
                            break
                        tmp += exp[i]
                        i += 1
                    ret.append(tmp)
                i += 1
            return (ret, i)
        self.Tree, _ = init(exp, 0)   


print(AutoTree("( pattern8 ((c6 BR) (d4 BR))(d6 e3 e4 e5 e6 f2 f3 f4 f5 f6 g1 g2 g3 g4 g5 g6)(c6 d4 BR)[(f3 [(pattern2ab (e3 e4) (d4 f3))][(pattern2ab (g2 g3) (f3 BR))])(e5 [(d6) (e4)][(pattern13 (e6 f4 f5 f6 g3 g4 g5 g6) (e5 BR))])(f2 [(pattern2ab (g1 g2) (f2 BR))][(pattern9 (g5 g4 f5 f4 f3 e5 e4 e3) (BR f2 d4))])(e3 [(pattern17 (d6 e5 e6 f2 f3 f4 f5 g1 g2 g3 g4 g5) (c6 d4 e3 BR))]) ])").Tree)
