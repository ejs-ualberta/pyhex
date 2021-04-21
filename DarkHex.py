
from copy import copy, deepcopy
import time
import math
from collections import deque
from hex_simple import Pattern
from hex_simple import UnionFind

from sgf_parse import SgfTree

PTS = '.xo'
EMPTY, BLACK, WHITE = 0, 1, 2
ECH, BCH, WCH = PTS[EMPTY], PTS[BLACK], PTS[WHITE]


def oppCH(ch):
  if ch== BCH: return WCH
  elif ch== WCH: return BCH
  else: assert(False)

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


'''
class Strategy:
  def __init__(self, ptm):
    self.ptm = ptm

  def update_strat(self, up_s):
    pass

  def best_response(self):
    # return best response strategy and its probability of winning
    pass
'''

class DarkHexBoard:
  def __init__(self, rows, cols):
    self.R, self.C, self.n = rows, cols, rows*cols
    self.brds = {BCH:ECH*self.n, WCH:ECH * self.n}
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

    self.connections = {BCH:UnionFind(), WCH:UnionFind()}
    self.connection_graphs = {BCH:self.get_connections(BCH, self.brds[BCH]), WCH:self.get_connections(WCH, self.brds[WCH])}

    self.dc_patterns = [
      #  x x
      # x *
      #    o
      Pattern([[0, 0, 0], [-1, 1, 0], [0, 1, -1], [1, 0, -1], [0, -1, 1]],
              [ECH, BCH, BCH, BCH, WCH], self.R, self.C),
      #  o o
      # o *
      #    x
      Pattern([[0, 0, 0], [-1, 1, 0], [0, 1, -1], [1, 0, -1], [0, -1, 1]],
              [ECH, WCH, WCH, WCH, BCH], self.R, self.C),
      #  x x
      # x * x
      Pattern([[0, 0, 0], [-1, 1, 0], [0, 1, -1], [1, 0, -1], [1, -1, 0]],
              [ECH, BCH, BCH, BCH, BCH], self.R, self.C),
      #  o o
      # o * o
      Pattern([[0, 0, 0], [-1, 1, 0], [0, 1, -1], [1, 0, -1], [1, -1, 0]],
              [ECH, WCH, WCH, WCH, WCH], self.R, self.C),
      #  x
      # x * o
      #    o
      Pattern([[0, 0, 0], [-1, 1, 0], [0, 1, -1], [0, -1, 1], [1, -1, 0]],
              [ECH, BCH, BCH, WCH, WCH], self.R, self.C),
    ]


  def dead(self, ptm):
    # Uses patterns to find dead cells.
    # Does not find all dead cells.
    inf_cs = set()
    for i in range(len(self.brds[ptm])):
      if self.brds[ptm][i] != ECH:
        continue
      for pat in self.dc_patterns:
        inf_cs = inf_cs.union(pat.matches(self.brds[ptm], i))
    return inf_cs

  def connected_cells(self, pt, ptm, brd, side1, side2):
    # Find all ptm-occupied cells connected to a particular ptm-occupied cell. Cells are connected if
    # there is a path between them of only ptm cells.
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    q, seen, reachable = deque([]), set(), set()
    if brd[pt] == ptm:
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
          if brd[n] == ptm and n not in seen:
            q.append(n)
          elif brd[n] == ECH:
            reachable.add(n)
    return seen, reachable

  def get_connections(self, ptm, brd):
    # Build the connection graphs
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    connections = {}

    # Connect adjacent empty cells, and create special side nodes
    side1 = self.n
    side2 = side1 + 1
    connections[side1] = set()
    connections[side2] = set()
    for i in range(self.n):
      if brd[i] == ECH:
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
          if brd[n] == ECH:
            connections[i].add(n)

    # Connect cells that are joined by ptm stones
    seen = set()
    for i in range(self.n):
      if brd[i] == ptm and i not in seen:
        s, r = self.connected_cells(i, ptm, brd, side1, side2)
        seen = seen.union(s)
        for c in r:
          cr = connections[c].union(r)
          cr.remove(c)
          connections[c] = cr
    return connections, side1, side2

  def shortest_path_len_from_to(self, connections, node, end):
    _, side1, side2 = self.connection_graphs[BCH]
    dists = [math.inf for i in range(self.n+2)]
    q = deque([node])
    dists[node] = 0
    while q:
      n = q.popleft()
      if n == end:
        break
      for n1 in connections[n]:
        d = dists[n] + 1
        if dists[n1] > d:
          dists[n1] = d
          q.append(n1)
    return dists[end] - (end in {side1, side2})

  def has_win(self, ptm):
    g, side1, side2 = self.connection_graphs[ptm]
    return side1 in g[side2]

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
    if ch != ECH and self.brds[ch][where] != ECH:
      print('sorry, position occupied')
      return ret
    return self.move(ch, where)

  def move(self, ch, where):
    brd = self.brds[ch]
    optm = oppCH(ch)
    if brd[where] != ECH:
      return False
    self.H.append((ch, where, copy(self.brds), deepcopy(self.connections), copy(self.connection_graphs)))
    ret = False
    if self.brds[optm][where] == optm:
      self.brds[ch] = change_str(brd, where, optm)
    else:
      self.brds[ch] = change_str(brd, where, ch)
      ret = True
    self.connection_graphs[ch] = self.get_connections(ch, self.brds[ch])
    return ret

  def move_to_brd(self, ch, where, brd_ch):
    obrd_ch = oppCH(brd_ch)
    brd = self.brds[brd_ch]
    if brd[where] != ECH:
      return False
    self.H.append((ch, where, copy(self.brds), deepcopy(self.connections), copy(self.connection_graphs)))
    self.brds[brd_ch] = change_str(brd, where, ch)
    self.connection_graphs[brd_ch] = self.get_connections(brd_ch, self.brds[brd_ch])
    self.connection_graphs[obrd_ch] = self.get_connections(obrd_ch, self.brds[brd_ch])
    return True

  def save_state(self):
    return (copy(self.brds), copy(self.connections), copy(self.connection_graphs))

  def restore_state(self, state):
    self.brds, self.connections, self.connection_graphs = state

  def refresh(self, brd_ch):
    obrd_ch = oppCH(brd_ch)
    self.connection_graphs[brd_ch] = self.get_connections(brd_ch, self.brds[brd_ch])
    self.connection_graphs[obrd_ch] = self.get_connections(obrd_ch, self.brds[brd_ch])

  
  '''
  def b_win_move(self, nbs=0, nws=0):
    calls = 1
    if self.has_win(BCH):
      return True, 1

    splen = self.shortest_path_len_from_to(*self.connection_graphs[WCH])
    if (splen <= (nbs - nws)):
      return '', 1

    moves = {i for i in range(self.n) if self.brds[BCH][i]==ECH}
    if nbs == nws:
      for move in moves:
        self.move_to_brd(BCH, move, BCH)
        wm, c = self.b_win_move(nbs + 1, nws)
        self.undo()
        calls += c
        if wm:
          return point_to_alphanum(move, self.C), calls
    else:
      for move in moves:
        self.move_to_brd(BCH, move, BCH)
        wm, c = self.b_win_move(nbs + 1, nws)
        self.undo()
        calls += c
        self.move_to_brd(WCH, move, BCH)
        wm1, c = self.b_win_move(nbs, nws + 1)
        self.undo()
        calls += c
        if wm and wm1:
          return point_to_alphanum(move, self.C), calls
    return '', calls

  def w_win_move(self, nbs=0, nws=0):
    calls = 1
    if self.has_win(WCH):
      return True, 1

    # splen is 1 + the length of the shortest path when going from side1 to side2 or vice versa
    splen = self.shortest_path_len_from_to(*self.connection_graphs[BCH])
    if (splen <= (nws - nbs) + 1):
      return '', 1

    moves = {i for i in range(self.n) if self.brds[WCH][i]==ECH}
    if nbs == nws + 1:
      for move in moves:
        self.move_to_brd(WCH, move, WCH)
        wm, c = self.w_win_move(nbs, nws + 1)
        self.undo()
        calls += c
        if wm:
          return point_to_alphanum(move, self.C), calls
    else:
      for move in moves:
        self.move_to_brd(WCH, move, WCH)
        wm, c = self.w_win_move(nbs, nws+1)
        self.undo()
        calls += c
        self.move_to_brd(BCH, move, WCH)
        wm1, c = self.w_win_move(nbs + 1, nws)
        self.undo()
        calls += c
        if wm and wm1:
          return point_to_alphanum(move, self.C), calls
    return '', calls

  def win_move(self, ptm):
    # assume neither player has won yet
    ret = ''
    cg = self.connection_graphs
    self.connection_graphs = {BCH:self.get_connections(BCH, ptm), WCH:self.get_connections(WCH, ptm)}
    if ptm == BCH:
      brd = self.brds[BCH]
      ret =  self.b_win_move(brd.count(BCH), brd.count(WCH))
    elif ptm == WCH:
      brd = self.brds[WCH]
      ret = self.w_win_move(brd.count(BCH), brd.count(WCH))
    self.connection_graphs = cg
    return ret
  
  def bound_win_prob(self, bstrat, wstrat, iterations=100):
    bs = deepcopy(bstrat)
    ws = deepcopy(wstrat)
    lb = 0.0
    ub = 0.0
    for i in range(iterations):
      br, np = bs.best_response()
      ws.update_strat(br)
      lb = max(1.0-np, lb)
      br, np = ws.best_response()
      ub = min(np, ub)
      bs.update_strat(br)
    return lb, ub
  '''

  def _win_move(self, ptm, hidden):
    optm = oppCH(ptm)
    calls = 1
    if self.has_win(ptm):
      return True, 1

    splen = self.shortest_path_len_from_to(*self.connection_graphs[optm])
    if (splen <= hidden):
      return '', 1

    dead = self.dead(ptm)
    state = self.save_state()
    for c in dead:
        self.brds[ptm] = change_str(self.brds[ptm], c, ptm)
    self.refresh(ptm)

    moves = {i for i in range(self.n) if self.brds[ptm][i]==ECH}
    if not hidden:
      for move in moves:
        self.move_to_brd(ptm, move, ptm)
        wm, c = self._win_move(ptm, hidden + 1)
        self.undo()
        calls += c
        if wm:
          self.restore_state(state)
          return point_to_alphanum(move, self.C), calls
    else:
      for move in moves:
        self.move_to_brd(ptm, move, ptm)
        wm, c = self._win_move(ptm, hidden + 1)
        self.undo()
        calls += c
        self.move_to_brd(optm, move, ptm)
        wm1, c = self._win_move(ptm, hidden - 1)
        self.undo()
        calls += c
        if wm and wm1:
          self.restore_state(state)
          return point_to_alphanum(move, self.C), calls

    self.restore_state(state)
    return '', calls

  def win_move(self, ptm, hidden=None):
    # assume neither player has won yet
    if hidden == None:
      optm = oppCH(ptm)
      hidden = self.brds[optm].count(optm) - self.brds[ptm].count(optm)
    ret = ''
    cg = self.connection_graphs
    self.connection_graphs = {BCH:self.get_connections(BCH, self.brds[ptm]), WCH:self.get_connections(WCH, self.brds[ptm])}
    ret, c = self._win_move(ptm, hidden)
    self.connection_graphs = cg
    return ret, c

  def find_winning_positions(self):
    # TODO: finish this
    BT, WT, NT = (0, 1, 2)
    def illegal(brd, occ):
      le = len(occ[ECH])
      lw = len(occ[WCH])
      lb = len(occ[BCH])
      # TODO: More illegal boards?
      if le < lb - lw - 1 or le < lw - lb: 
        return True
      return False

    def init_states(layers, brd):
      occ = {BCH:[], WCH:[], ECH:[]}
      for i in range(self.n):
        occ[brd[i]].append(i)
      if brd in layers[len(occ[ECH])] or illegal(brd, occ):
        return

      bconn, side1, side2 = self.get_connections(BCH, brd)
      wconn, _, _ = self.get_connections(WCH, brd)
      if side1 in bconn[side2]:
        layers[len(occ[ECH])][brd] = BT
      elif side1 in wconn[side2]:
        layers[len(occ[ECH])][brd] = WT
      else:
        layers[len(occ[ECH])][brd] = NT

      for move in occ[ECH]:
        init_states(layers, change_str(brd, move, BCH))
        init_states(layers, change_str(brd, move, WCH))

    empty = [i for i in range(self.n) if self.brds[BCH][i] == self.brds[WCH][i] == ECH]
    layers = [{} for i in range(len(empty) + 1)]
    init_states(layers, ECH * self.n)

    i = 0
    wins = {BT:[], WT:[]}
    for layer in layers[1:]:
      for state in layer:
        for move in empty:
          s1 = change_str(state, move, BCH)
          s2 = change_str(state, move, WCH)
          if s1 not in layers[i] or s2 not in layers[i]:
            continue
          li1 = layers[i][s1]
          li2 = layers[i][s2]


  def showboard(self, ptm):
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
        pretty += ' ' + paint([self.brds[ptm][coord_to_point(j,k,self.C)]]) 
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
      print('original position, nothing to undo')
      return False
    else:
        _, _, self.brds, self.connections, self.connection_graphs = self.H.pop()
    return True

  def msg(self, ch, hidden):
    optm = oppCH(ch)
    if self.has_win(ch): return(ch + ' has won')
    elif self.has_win(optm): return(optm + ' has won')
    else:
      st = time.time()
      wm, calls = self.win_move(ch, hidden)
      out = '\n' + ch + '-to-move: '
      out += (ch + ' wins' if wm else ch + ' does not win with probability 1')
      out += (' ... ' if wm else ' ') + wm + '\n'
      out += str(calls) + ' calls\n'
      out += "%.4f" % (time.time() - st) + 's \n'
      return out

  def save(self, fname):
    header = "(;FF[4]GM[11]SZ[%d:%d]\n" % (self.C, self.R)
    chrs = {BCH:'B', WCH:'W'}
    sgf = ""
    for i in range(len(self.H)-1, -1, -1):
      h = self.H[i]
      ch = h[0]
      loc = point_to_alphanum(h[1], self.C)
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
  print('  h                                      help menu')
  print('  s x|o                             show the board')
  print('  ? x|o [#hidden]    solve the position for x or o')
  print('  z n m               change the board size to nxm')
  print('  x b2                                  play x b 2')
  print('  o e3                                  play o e 3')
  #print('  . a2                                   erase a 2')
  print('  pr x|o e3              play and reveal x|o at e3')
  print('  u                                           undo')
  print('  sv                          save the game as sgf')
  print('  ld                          load a game from sgf')
  print('  [return]                                    quit')


def interact():
  print()
  p = DarkHexBoard(3, 3)
  while True:
    cmd = input('> ').split()
    if len(cmd)==0:
      print('\n ... adios :)\n')
      return

    if cmd[0]=='h':
      printmenu()

    elif cmd[0]=='z':
      try:
        n = int(cmd[1])
        m = int(cmd[2])
        if (n > 1 and m > 1):
          p = DarkHexBoard(n, m)
      except:
        print("Command requires two natural numbers > 1.")

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
      if len(cmd) in {3, 2} and cmd[1] in {BCH, WCH}:
        hidden = None
        try:
          h = int(cmd[2])
          if h < 0:
            print("Please enter a number >= 0")
          else:
            hidden = h
        except:
          pass
        print(p.msg(cmd[1], hidden))


    elif (cmd[0] in PTS):
      print(["Move Unsuccessful", "Move Successful"][p.requestmove(cmd[0] + ' ' + ''.join(cmd[1:]))])

    elif (cmd[0]) == 'pr':
      success = 0
      try:
        ptm = cmd[1]
        optm = oppCH(ptm)
        success = p.requestmove(ptm + ' ' + ''.join(cmd[2:]))
        if success:
          p.requestmove(optm + ' ' + ''.join(cmd[2:]))
        print(["Move Unsuccessful", "Move Successful"][success])
      except:
        pass

    elif (cmd[0] == 's'):
      try:
        p.showboard(cmd[1])
      except:
        print("Please enter a valid player to show the board for.")

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
          p = DarkHexBoard(r, r)
        elif len(sz) == 2:
          p = DarkHexBoard(int(sz[1]), int(sz[0]))
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
