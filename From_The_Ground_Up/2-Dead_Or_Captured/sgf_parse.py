# _Simple_ sgf parser. For now it does not do very much.
# TODO: Make this better

whitespace = {' ', '\n', '\t'}
class SgfTree:
    def __init__(self, input_str=""):
        self.children = []
        self.properties = {}
        if len(input_str) >= 3 and input_str[0]=='(' and input_str[-1]==')':
            self.parse(input_str, 1, True)
        elif input_str:
            raise Exception("Invalid SGF.")

    def expect_alpha(self, s, i):
        out = ""
        while i < len(s):
            ch = s[i]
            if not ch.isalpha():
                break
            i += 1
            out += ch
        if not out:
            raise Exception("Expected alpha characters.")
        return i, out

    # This also accepts colons so numberpairs/pointpairs can be entered
    def expect_pp_or_alnum(self, s, i):
        out = ""
        while i < len(s):
            ch = s[i]
            if not ch.isalnum() and ch != ':':
                break
            i += 1
            out += ch
        if not out:
            raise Exception("Expected alphanum string.")
        return i, out

    def ignore_ws(self, s, i):
        while i < len(s):
            ch = s[i]
            if ch in whitespace:
                i += 1
                continue
            break
        return i

    def parse(self, sgf_str, i, paren):
        if i >= len(sgf_str):
            raise Exception("Parse error")
        while i < len(sgf_str):
            ch = sgf_str[i]
            if ch in whitespace:
                i = self.ignore_ws(sgf_str, i)
            elif ch == ';':
                node = SgfTree()
                self.children.append(node)
                i = self.ignore_ws(sgf_str, i)
                i = node.parse(sgf_str, i+1, False)
            elif ch == '(':
                i = self.parse(sgf_str, i+1, True)
            elif ch == ')':
                if paren:
                    i += 1
                return i
            elif ch.isalpha():
                i, prop = self.expect_alpha(sgf_str, i)
                i = self.ignore_ws(sgf_str, i)
                if i >= len(sgf_str) or sgf_str[i] != '[':
                    raise Exception("Property missing value/s.")
                values = []
                while i < len(sgf_str) and sgf_str[i] == '[':
                    i += 1
                    i = self.ignore_ws(sgf_str, i)
                    # This doesn't switch based on prop so right now it
                    # matches numberpairs/pointpairs and alphanumeric text without whitespace
                    i, val = self.expect_pp_or_alnum(sgf_str, i)
                    values.append(val)
                    i = self.ignore_ws(sgf_str, i)
                    if i >= len(sgf_str) or sgf_str[i] != ']':
                        raise Exception("Unclosed property value.")
                    i += 1
                    i = self.ignore_ws(sgf_str, i)
                self.properties[prop] = values
        return i
