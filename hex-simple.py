"""
negamax small-board hex solver

based on ttt and 3x3 go programs,
special move order for 3x3, 3x4, 4x4 only,
too slow for larger boards

4x4 empty board, x-to-move, x wins, 7034997 calls
"""

import numpy as np
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
    self.cache = dict()
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
    #elif self.R == 5 and self.C == 5: self.CELLS = (13, 14, 12, 15, 11, 16, 10, 17, 9, 18, 8, 19, 7, 20, 6, 21, 5, 22, 4, 23, 3, 24, 2, 1)
    else: self.CELLS = [j for j in range(self.n)]  # this order terrible for solving

  def requestmove(self, cmd):
    c = cmd
    parseok, cmd = False, cmd.split()
    ret = ''
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
        
  def win_move(self, ptm): # assume neither player has won yet
    tt = self.cache
    blanks, calls = [], 1
    for j in self.CELLS:
      if self.brd[j]==ECH: blanks.append(j)
    optm = oppCH(ptm)
    for k in blanks:
      self.brd = change_str(self.brd, k, ptm)
      self.H.append((ptm, k))
      if self.brd in tt and tt[self.brd][1] == ptm:
        ret = (tt[self.brd][0], 1)
        self.undo()
        return ret
      if self.has_win(ptm):
        pt = point_to_alphanum(k, self.C)
        tt[self.brd] = (pt, ptm)
        self.undo()
        return pt, calls
      cw, prev_calls = self.win_move(optm)
      calls += prev_calls
      if not cw:
        pt = point_to_alphanum(k, self.C)
        tt[self.brd] = (pt, ptm)
        self.undo()
        return pt, calls
      self.undo()
    return '', calls

  def showboard(self):
    def paint(s):
      pt = ''
      for j in s:
        if j in PTS:      pt += stonecolors[PTS.find(j)] + j + colorend
        elif j.isalnum(): pt += textcolor + j + colorend
        else:             pt += j
      return pt

    pretty = '\n   ' 
    for c in range(self.C): # columns
      pretty += ' ' + paint(chr(ord('a')+c))
    pretty += '\n'
    for j in range(self.R): # rows
      pretty += ' ' + ' '*j + paint(str(1+j)) + ' '
      for k in range(self.C): # columns
        pretty += ' ' + paint([self.brd[coord_to_point(j,k,self.C)]])
      pretty += '\n'
    print(pretty)

  def undo(self):  # pop last meta-move
    if not self.H:
      print('\n    original position,  nothing to undo\n')
    else:
      self.brd = change_str(self.brd, self.H.pop()[1], ECH)

  def msg(self, ch):
    if self.has_win('x'): return('x has won')
    elif self.has_win('o'): return('o has won')
    else: 
      wm, calls = self.win_move(ch)
      out = '\n' + ch + '-to-move: '
      out += (ch if wm else oppCH(ch)) + ' wins' 
      out += (' ... ' if wm else ' ') + wm + '\n'
      out += str(calls) + ' calls\n'
      return out


def printmenu():
  print('  h             help menu')
  print('  z        show the board')
  print('  x b2         play x b 2')
  print('  o e3         play o e 3')
  print('  . a2          erase a 2')
  print('  u                  undo')
  print('  [return]           quit')


def interact():
  p = Position(4, 4) #TODO: put x/o along sides
  history = []  # board positions
  while True:
    p.showboard()
    cmd = input(' ')
    if len(cmd)==0:
      print('\n ... adios :)\n')
      return
    if cmd[0][0]=='h':
      printmenu()
    elif cmd[0][0]=='z':
      cmd = cmd.split()
      try:
        sz = int(cmd[1])
        if (sz > 0):
          p = Position(sz, sz)
      except:
        pass
    elif cmd[0][0]=='u':
      p.undo()
    elif cmd[0][0]=='?':
      cmd = cmd.split()
      if len(cmd)>0:
        if cmd[1][0]=='x': 
          print(p.msg('x'))
        elif cmd[1][0]=='o': 
          print(p.msg('o'))
    elif (cmd[0][0] in PTS):
      p.requestmove(cmd)

interact()
