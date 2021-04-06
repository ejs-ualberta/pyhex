"""
negamax small-board hex solver
"""
import time
from collections import deque

from sgf_parse import SgfTree

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
  return 'abcdefghi'[c] + '123456789'[r]

def alphanum_to_point(an, C):
  return (ord(an[0]) - ord('a')) + (int(an[1:])-1)*C

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
        self.nbrs.append(set(nbs))

    self.LFT_COL, self.RGT_COL, self.TOP_ROW, self.BTM_ROW = set(), set(), set(), set()
    for r in range(self.R):
      self.LFT_COL.add(coord_to_point(r, 0, self.C))
      self.RGT_COL.add(coord_to_point(r, self.C-1, self.C))
    for c in range(self.C):
      self.TOP_ROW.add(coord_to_point(0, c, self.C))
      self.BTM_ROW.add(coord_to_point(self.R-1, c, self.C))

    self.connection_graphs = {BCH:self.get_connections(BCH), WCH:self.get_connections(WCH)}

  def requestmove(self, cmd):
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
    self.move(ch, where)
    return True

  def move(self, ch, where):
    self.H.append((self.brd[where], where, self.connection_graphs))
    self.brd = change_str(self.brd, where, ch)
    self.connection_graphs = {BCH:self.get_connections(BCH), WCH:self.get_connections(WCH)}

  def connected_cells(self, pt, ptm, side1, side2):
    # Find all ptm-occupied cells connected to a particular ptm-occupied cell. Cells are connected if
    # there is a path between them of only ptm cells.
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    q, seen, reachable = deque([]), set(), set()
    if self.brd[pt] == ptm:
      seen = {pt}
      q.append(pt)
      while len(q) > 0:
        c = q.popleft()
        seen.add(c)
        if c in set1:
          reachable.add(side1)
        elif c in set2:
          reachable.add(side2)
        nbrs = self.nbrs[c]
        for n in nbrs:
          if self.brd[n] == ptm and n not in seen:
            q.append(n)
          elif self.brd[n] == ECH:
            reachable.add(n)
    return seen, reachable

  def get_connections(self, ptm):
    # Build the connection graphs
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    connections = {}

    # Connect adjacent empty cells, and create special side nodes
    side1 = len(self.brd)
    side2 = side1 + 1
    connections[side1] = set()
    connections[side2] = set()
    for i in range(len(self.brd)):
      if self.brd[i] == ECH:
        nbrs = self.nbrs[i]
        connections[i] = set()
        # Connect to two "sides"
        if i in set1:
          connections[side1].add(i)
          connections[i].add(side1)
        elif i in set2:
          connections[i].add(side2)
          connections[side2].add(i)
        # Connect adjacent empty cells
        for n in nbrs:
          if self.brd[n] == ECH:
            connections[i].add(n)

    # Connect cells that are joined by ptm stones
    seen = set()
    for i in range(len(self.brd)):
      if self.brd[i] == ptm and i not in seen:
        s, r = self.connected_cells(i, ptm, side1, side2)
        seen = seen.union(s)
        for c in r:
          cr = connections[c].union(r)
          cr.remove(c)
          connections[c] = cr
    return connections, side1, side2

  def win_move(self, ptm):
    # assume neither player has won yet
    optm = oppCH(ptm) 
    calls = 1
    ovc = set()

    mustplay = {i for i in range(len(self.brd)) if self.brd[i] == ECH}

    while len(mustplay) > 0:
      move = next(iter(mustplay)) # This is not a good ordering.

      self.move(ptm, move)
      g, side1, side2 = self.connection_graphs[ptm]

      if side1 in g[side2]:
        # There is a connection between the two sides of the current player
        pt = point_to_alphanum(move, self.C)
        self.undo()
        return pt, calls, {move}

      omv, ocalls, oset = self.win_move(optm)

      calls += ocalls
      if not omv: # opponent has no winning response to ptm move
        oset.add(move)
        pt = point_to_alphanum(move, self.C)
        self.undo()
        return pt, calls, oset

      ovc = ovc.union(oset)
      mustplay = mustplay.intersection(oset)
      self.undo()

    return '', calls, ovc

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

  def show_big_board(self, info, colours, spc='  '):
    # info is a list with the same length as the board that contains what to place at each cell.
    # colours is a list the same length as info. If an entry is None, use the default colour, otherwise use the colour in the entry.
    def paint(s, colour=None):
      if colour:
        pt = ''
        for j in s:
          pt += colour + j + colorend
        return pt
      return s

    hs = ' ' * (len(spc)//2)
    rem = len(spc)%2
    pretty = '\n  ' + hs + rem*' '
    for c in range(self.C): # columns
      pretty += spc + paint(chr(ord('a')+c), textcolor)
    pretty += '\n'*(len(spc)//2)
    pretty += '\n '+ spc + '+' + spc
    for c in range(self.C): # columns
      pretty += paint(BCH, stonecolors[BLACK]) + spc
    pretty += '+\n'
    pretty += '\n'*(len(spc)//2)
    for j in range(self.R): # rows
      n = str(1+j)
      pretty += spc*(j//2) + hs*(j%2) + ' '*rem + ' '*(len(hs)+rem) + paint(n, textcolor) + ' '*(len(spc)-len(n)+1) + paint(WCH, stonecolors[WHITE]) + ' '*(len(hs)+rem)
      for k in range(self.C): # columns
        item = info[coord_to_point(j,k,self.C)]
        pretty += ' '*(len(hs) - len(item)//2) + paint(item, colours[coord_to_point(j,k,self.C)]) + ' '*(len(hs)+rem - len(item)//2 - len(item)%2 + 1)
      pretty += hs + paint(WCH, stonecolors[WHITE]) + '\n'
      pretty += '\n'*(len(spc)//2)

    pretty += spc*(self.R//2) + hs*(self.R%2) + ' '*rem + ' '*(len(hs)+rem) + spc + ' ' + '+'
    for c in range(self.C):
      pretty += spc + paint(BCH, stonecolors[BLACK])
    pretty += spc + '+\n'
    print(pretty)


  def undo(self):  # pop last meta-move
    if not self.H:
      print('\n    original position,  nothing to undo\n')
      return False
    else:
      ch, where, self.connection_graphs = self.H.pop()
      self.brd = change_str(self.brd, where, ch)
    return True

  def msg(self, ch):
    optm = oppCH(ch)
    gptm, pside1, pside2 = self.connection_graphs[ch]
    goptm, oside1, oside2 = self.connection_graphs[optm]
    if pside1 in gptm[pside2]: return(ch + ' has won')
    elif oside1 in goptm[oside2]: return(optm + ' has won')
    else:
      st = time.time()
      wm, calls, vc = self.win_move(ch)
      out = '\n' + ch + '-to-move: '
      out += (ch if wm else oppCH(ch)) + ' wins' 
      out += (' ... ' if wm else ' ') + wm + '\n'
      out += str(calls) + ' calls\n'
      out += "%.4f" % (time.time() - st) + 's \n'
      return out

  def save(self, fname):
    header = "(;FF[4]GM[11]SZ[%d:%d]\n" % (self.C, self.R)
    chrs = {BCH:'B', WCH:'W'}
    visited = set()
    sgf = ""
    for i in range(len(self.H)-1, -1, -1):
      h = self.H[i]
      pt = h[1]
      loc = point_to_alphanum(pt, self.C)
      ch = self.brd[pt]
      if ch == ECH or pt in visited:
        continue
      visited.add(pt)
      sgf = ';' + chrs[ch] + '[' + loc + ']' + sgf
    sgf += '\n)'
    f = None
    try:
      f = open(fname, 'w')
    except:
      print("Bad filename")
      return
    f.write(header + sgf)
    f.close()

def printmenu():
  print('  h                              help menu')
  print('  s                         show the board')
  print('  ? x|o      solve the position for x or o')
  print('  z                         show the board')
  print('  x b2                          play x b 2')
  print('  o e3                          play o e 3')
  print('  . a2                           erase a 2')
  print('  u                                   undo')
  print('  sv                  save the game as sgf')
  print('  ld                  load a game from sgf')
  print('  [return]                            quit')


def interact():
  print()
  p = Position(4, 4)
  while True:
    cmd = input('> ').split()
    if len(cmd)==0:
      print('\n ... adios :)\n')
      return

    if cmd[0]=='h':
      printmenu()

    elif cmd[0]=='z':
      try:
        sz = int(cmd[1])
        if (sz > 1):
          p = Position(sz, sz)
      except:
        print("Command requires natural number > 1.")

    elif cmd[0]=='u':
      if len(cmd) == 2:
        try:
          for i in range(int(cmd[1])):
            if not p.undo(): break
        except:
          print("Not a natural number.")
      else:
        p.undo()

    elif cmd[0]=='?':
      if len(cmd) > 0:
        if cmd[1] in {BCH, WCH}: 
          print(p.msg(cmd[1]))

    elif (cmd[0] in PTS):
      p.requestmove(cmd[0] + ' ' + ''.join(cmd[1:]))

    elif (cmd[0] == 's'):
      p.showboard()

    elif cmd[0] == 'sv':
      if len(cmd) != 2:
        print("Please enter a valid file name.")
        continue
      p.save(cmd[1])

    elif cmd[0] == 'ld':
      if len(cmd) != 2:
        print("Please enter a valid file name.")
        continue
      fname = cmd[1]
      f = None
      try:
        f = open(fname, 'r')              
      except:
        print("Failed to open file.")
        continue

      t = None
      try:
        t = SgfTree(f.read())
        sz = t.children[0].properties["SZ"][0].split(':')
        if len(sz) == 1:
          r = int(sz[0])
          p = Position(r, r)
        elif len(sz) == 2:
          p = Position(int(sz[1]), int(sz[0]))
        else:
          raise Exception("Bad")

        while t.children:
          t1 = t.children[0]
          props = t1.properties
          chrs = {"B":BCH, "W":WCH}
          for ch in chrs:
            if ch in props:
              p.move(chrs[ch], alphanum_to_point(props[ch][0], p.C))
          t = t1
      except:
        print("Failed to parse sgf.")
      f.close()

    else:
      print("Unknown command.")


if __name__ == "__main__":
  interact()

#interact()
