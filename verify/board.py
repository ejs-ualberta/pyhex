"""
negamax small-board hex solver

based on ttt and 3x3 go programs,
special move order for 3x3, 3x4, 4x4 only,
too slow for larger boards

4x4 empty board, x-to-move, x wins, 7034997 calls
"""
from copy import copy
from collections import deque

"""
points on the board
"""

PTS = '.xo'
EMPTY, BLACK, WHITE = 0, 1, 2
ECH, BCH, WCH = PTS[EMPTY], PTS[BLACK], PTS[WHITE]

def oppCH(ch): 
  if ch== BCH: return WCH
  elif ch== WCH: return BCH
  else: assert(False)

"""
board: one-dimensional string

index positions for     board:    0 1 2       <- row 0
                                   3 4 5       <- row 1
                                    6 7 8       <- row 2
"""

def coord_to_point(r, c, C): 
  return c + r*C

def point_to_coord(p, C): 
  return divmod(p, C)

def point_to_alphanum(p, C):
  r, c = point_to_coord(p, C)
  return 'abcdefghj'[c] + '1234566789'[r]

def pointset_to_str(S):
  s = ''
  for j in range(N):
    s += BCH if j in S else ECH
  return s

def change_str(s, where, what):
  return s[:where] + what + s[where+1:]

def char_to_color(c): 
  return PTS.index(c)

escape_ch           = '\033['
colorend, textcolor = escape_ch + '0m', escape_ch + '0;37m'
stonecolors         = (textcolor, escape_ch + '0;35m', escape_ch + '0;32m')


class Position: # hex board 
  def __init__(self, rows, cols):
    self.R, self.C, self.n = rows, cols, rows*cols
    self.brd = PTS[EMPTY]*self.n
    self.H = []
    #self.cache = dict()

    self.nbrs = []
    for r in range(self.R):
      for c in range(self.C):
        nbs = []
        if r > 0:                  nbs.append(coord_to_point(r-1, c,   self.C))
        if r > 0 and c < self.C-1: nbs.append(coord_to_point(r-1, c+1, self.C))
        if c > 0:                  nbs.append(coord_to_point(r,   c-1, self.C))
        if c < self.C-1:           nbs.append(coord_to_point(r,   c+1, self.C))
        if r < self.R-1 and c > 0: nbs.append(coord_to_point(r+1, c-1, self.C))
        if r < self.R-1:           nbs.append(coord_to_point(r+1, c,   self.C))
        self.nbrs.append(nbs)

    self.LFT_COL, self.RGT_COL, self.TOP_ROW, self.BTM_ROW = set(), set(), set(), set()
    for r in range(self.R):
      self.LFT_COL.add(coord_to_point(r, 0, self.C))
      self.RGT_COL.add(coord_to_point(r, self.C-1, self.C))
    for c in range(self.C):
      self.TOP_ROW.add(coord_to_point(0, c, self.C))
      self.BTM_ROW.add(coord_to_point(self.R-1, c, self.C))

    if self.R == 3 and self.C == 3: self.CELLS = (4,2,6,3,5,1,7,0,8)
    elif self.R == 3 and self.C == 4: self.CELLS = (5,6,4,7,2,9,3,8,1,10,0,11)
    elif self.R == 4 and self.C == 4: self.CELLS = (6,9,3,12,2,13,5,10,8,7,1,14,4,11,0,15)
    elif self.R == 5 and self.C == 5: 
      self.CELLS = (12,8,16,7,17,6,18,11,13,4,20,3,21,2,22,15,9,10,14,5,19,1,23,0,24)
    else: self.CELLS = [j for j in range(self.n)]  # this order terrible for solving

  def requestmove(self, cmd):
    c = cmd
    ret, cmd = False, cmd.split()
    if len(cmd) != 2:
      print('invalid command')
      return ret
    ch = cmd[0][0]
    if ch not in PTS:
      print('bad character')
      return ret
    q, n = cmd[1][0], cmd[1][1:]
    if (not q.isalpha()) or (not n.isdigit()):
      print('not alphanumeric')
      return ret
    x, y = int(n) - 1, ord(q)-ord('a')
    if x<0 or x >= self.R or y<0 or y >= self.C:
      print('coordinate off board')
      return ret
    where = coord_to_point(x,y,self.C)
    if ch != ECH and self.brd[where] != ECH:
      print('\n  sorry, position occupied')
      return ret
    self.brd = change_str(self.brd, where, ch)
    if ch != ECH:
      self.H.append((ch, where))
    return True

  def has_win(self, who):
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if who == BCH else (self.LFT_COL, self.RGT_COL)
    Q, seen = deque([]), set()
    for c in set1:
      if self.brd[c] == who: 
        Q.append(c)
        seen.add(c)
    while len(Q) > 0:
      c = Q.popleft()
      if c in set2: 
        return True
      for d in self.nbrs[c]:
        if self.brd[d] == who and d not in seen:
          Q.append(d)
          seen.add(d)
    return False


  def showboard(self):
    def paint(s):
      pt = ''
      for j in s:
        if j in PTS:      pt += stonecolors[PTS.find(j)] + j + colorend
        elif j.isalnum(): pt += textcolor + j + colorend
        else:             pt += j
      return pt

    pretty = '\n '
    for c in range(self.C): # columns
      pretty += ' ' + paint(chr(ord('a')+c))
    pretty += '\n + '
    for c in range(self.C): # columns
      pretty += paint(BCH) + ' '
    pretty += '+\n'
    for j in range(self.R): # rows
      pretty += ' '*j + paint(str(1+j)) + ' ' + paint(WCH)
      for k in range(self.C): # columns
        pretty += ' ' + paint([self.brd[coord_to_point(j,k,self.C)]]) 
      pretty += ' ' + paint(WCH) + '\n'

    pretty += '  '  + ' ' * self.R + '+'
    for c in range(self.C):
      pretty += ' ' + paint(BCH)
    pretty += ' +\n'
    print(pretty)


  def undo(self):  # pop last meta-move
    if not self.H:
      print('\n    original position,  nothing to undo\n')
    else:
      self.brd = change_str(self.brd, self.H.pop()[1], ECH)
