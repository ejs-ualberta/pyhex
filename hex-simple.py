"""
negamax small-board hex solver
"""
from copy import copy, deepcopy
import time
import math
from collections import deque
import numpy as np

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

def point_to_cubic(pt, C):
  z, x = divmod(pt, C)
  return np.array([x, -x-z, z])

def cubic_to_point(vec, C):
  return vec[2] * C + vec[0]

def cubic_rotate_60_cc(vec):
  return np.array([-vec[1], -vec[2], -vec[0]])

def coord_to_point(r, c, C): 
  return c + r*C

def point_to_coord(p, C): 
  return divmod(p, C)

def point_to_alphanum(p, C):
  r, c = point_to_coord(p, C)
  return 'abcdefghi'[c] + '123456789'[r]

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


class UnionFind:
  def __init__(self):
    self.parents = {}

  def union(self, elem1, elem2):
    e1 = self.find(elem1)
    e2 = self.find(elem2)
    if e1 == e2:
      return
    self.parents[e2] = e1;

  def find(self, elem):
    while elem in self.parents:
      p = self.parents[elem]
      if p in self.parents:
        self.parents[elem] = self.parents[p]
      elem = p
    return elem;


class Pattern:
  # Represents a pattern of cells to match to cells on the board, could be a captured pattern or a dead cell pattern
  def __init__(self, offsets, chars, rows, cols):
    # Offsets are vectors representing offsets from the main cell of the pattern at [0, 0, 0]
    # chars must be in the same order as offsets, as in offsets[0] and chars[0] must describe the same cell.
    # If a char in chars == ECH it is treated as a cell that would be dead or captured if the pattern matches.
    self.offsets = offsets
    self.deltas = [self.offsets]
    self.chars = chars;
    self.R = rows
    self.C = cols

    #Precompute all rotations of the pattern, store them in deltas
    prev = self.deltas[-1]
    for i in range(6):
      self.deltas.append([cubic_rotate_60_cc(v) for v in prev])
      prev = self.deltas[-1]

  def matches(self, board, pt):
    # Convert the point on the board to cubic coordinates
    vec = point_to_cubic(pt, self.C)
    # Try each rotation of the pattern at pt
    ret = set()
    for rot in self.deltas:
      is_c = True
      for i in range(len(rot)):
        c = vec + rot[i]
        x, z = c[0], c[2]
        # If the point is off the board, check if it is on an edge of the same char
        if x < 0 or z < 0 or x >= self.C or z >= self.R:
          if self.on_correct_edge(c, self.chars[i]):
            continue
          is_c = False
          break
          continue
        # If the point is on the board, check if that point contains the correct character
        ch = board[cubic_to_point(c, self.C)]
        if ch != self.chars[i]:
          is_c = False
          break
      # If a pattern match is found, add the empty cells of the pattern to ret.
      if is_c:
        for i in range(len(self.chars)):
          if self.chars[i] == ECH:
            ret.add(cubic_to_point(vec + rot[i], self.C))
    return ret

  def on_correct_edge(self, vec, ch):
    # Check if a cell is on an edge that matches its colour
    x = vec[0]
    z = vec[2]
    # If a BCH cell has an x coord between 0 and self.C then it must be
    # outside the board on the z axis because of where this function is called.
    if ch == BCH:
      if 0 <= x and x < self.C:
        return True
    elif ch == WCH:
      if 0 <= z and z < self.R:
        return True
    # Empty cells do not match any edge
    # Obtuse corners match any colour
    return (x == self.C and z == -1) or (x == -1 and z == self.R)

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

    #if self.R == 3 and self.C == 3: self.CELLS = (4,2,6,3,5,1,7,0,8)
    #elif self.R == 3 and self.C == 4: self.CELLS = (5,6,4,7,2,9,3,8,1,10,0,11)
    #elif self.R == 4 and self.C == 4: self.CELLS = (6,9,3,12,2,13,5,10,8,7,1,14,4,11,0,15)
    #elif self.R == 5 and self.C == 5: 
    #  self.CELLS = (12,8,16,7,17,6,18,11,13,4,20,3,21,2,22,15,9,10,14,5,19,1,23,0,24)
    #else: self.CELLS = [j for j in range(self.n)]  # this order terrible for solving

    self.miai_patterns = {BCH:Pattern([np.array([0, 0, 0]), np.array([1, 1, -2]), np.array([0, 1, -1]), np.array([1, 0, -1])],
                                      [BCH, BCH, ECH, ECH], self.R, self.C),
                          WCH:Pattern([np.array([0, 0, 0]), np.array([1, 1, -2]), np.array([0, 1, -1]), np.array([1, 0, -1])],
                                      [WCH, WCH, ECH, ECH], self.R, self.C)}
    self.connection_graphs = {BCH:self.get_connections(BCH), WCH:self.get_connections(WCH)}
    self.miai_reply = self.get_miai_replies()
    self.miai_connections = {BCH:UnionFind(), WCH:UnionFind()}

    # dead cell patterns
    self.dc_patterns = [
      #  x x
      # x *
      #    o
      Pattern([np.array([0, 0, 0]), np.array([-1, 1, 0]), np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([0, -1, 1])],
              [ECH, BCH, BCH, BCH, WCH], self.R, self.C),
      #  o o
      # o *
      #    x
      Pattern([np.array([0, 0, 0]), np.array([-1, 1, 0]), np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([0, -1, 1])],
              [ECH, WCH, WCH, WCH, BCH], self.R, self.C),
      #  x x
      # x * x
      Pattern([np.array([0, 0, 0]), np.array([-1, 1, 0]), np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([1, -1, 0])],
              [ECH, BCH, BCH, BCH, BCH], self.R, self.C),
      #  o o
      # o * o
      Pattern([np.array([0, 0, 0]), np.array([-1, 1, 0]), np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([1, -1, 0])],
              [ECH, WCH, WCH, WCH, WCH], self.R, self.C),
    ]
    # black captured patterns
    self.bc_patterns = [
      # x x x
      #  * *
      #   x
      Pattern([np.array([0, 0, 0]), np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([0, 2, -2]), np.array([1, 1, -2]), np.array([2, 0, -2])],
              [BCH, ECH, ECH, BCH, BCH, BCH], self.R, self.C),
      #   x x
      #  * *
      # x x
      #Pattern([np.array([0, 0, 0]), np.array([1, -1, 0]), np.array([1, 0, -1]), np.array([2, -1, -1]), np.array([2, 0, -2]), np.array([3, -1, -2])],
              #[BCH, BCH, ECH, ECH, BCH, BCH], self.R, self.C)
    ]

    # white captured patterns
    self.wc_patterns = [
      # o o o
      #  * *
      #   o
      Pattern([np.array([0, 0, 0]), np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([0, 2, -2]), np.array([1, 1, -2]), np.array([2, 0, -2])],
              [WCH, ECH, ECH, WCH, WCH, WCH], self.R, self.C),
      #   o o
      #  * *
      # o o
      #Pattern([np.array([0, 0, 0]), np.array([1, -1, 0]), np.array([1, 0, -1]), np.array([2, -1, -1]), np.array([2, 0, -2]), np.array([3, -1, -2])],
              #[WCH, WCH, ECH, ECH, WCH, WCH], self.R, self.C)
    ]

  def miai_connected(self, ptm):
    # Check efficiently whether ptm stones are miai connected
    c = self.miai_connections[ptm]
    conn, side1, side2 = self.connection_graphs[ptm]
    if c.find(side1) == c.find(side2):
      return True
    return False

  def get_miai_connections(self, ptm):
    # Calculate from scratch whether ptm stones are miai connected
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    conn, side1, side2 = self.connection_graphs[ptm]
    c = UnionFind()
    for i in range(len(self.brd)):
      ch = self.brd[i]
      if ch != ptm:
        continue

      for nbr in self.nbrs[i]:
        if self.brd[nbr] == ptm:
          c.union(nbr, i)
      if i in set1:
        c.union(side1, i)
      elif i in set2:
        c.union(side2, i)
      cells = self.miai_patterns[ptm].matches(self.brd, i)
      for cell in cells:
        if cell in set1:
          c.union(side1, cell)
        elif cell in set2:
          c.union(side2, cell)
        c.union(i, cell)
    #print(c.find(side1), c.find(side2))
    return c

  def update_miai_at(self, miai_replies, idx):
    # Adds any new miai at self.brd[idx] to miai_replies
    ch = self.brd[idx]
    cells = self.miai_patterns[ch].matches(self.brd, idx)
    for cell in cells:
      nbrs = cells.intersection(set(self.nbrs[cell]))
      miai_replies[ch][cell] = miai_replies[ch][cell].union(nbrs)

  def get_miai_replies(self):
    miai_replies = {BCH:[set() for i in range(len(self.brd))], WCH:[set() for i in range(len(self.brd))]}
    for i in range(len(self.brd)):
      if self.brd[i] != ECH:
        self.update_miai_at(miai_replies, i)
    return miai_replies

  def update_miai_connections(self, ptm, i):
    optm = oppCH(ptm)
    # If ptm played in the opponents's miai
    if self.miai_reply[optm][i]:
      self.miai_connections[optm] = self.get_miai_connections(optm)

    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    conn, side1, side2 = self.connection_graphs[ptm]
    c = self.miai_connections[ptm]

    for nbr in self.nbrs[i]:
      if self.brd[nbr] == ptm:
        c.union(nbr, i)
    if i in set1:
      c.union(side1, i)
    elif i in set2:
      c.union(side2, i)
    cells = self.miai_patterns[ptm].matches(self.brd, i)
    for cell in cells:
      if cell in set1:
        c.union(side1, cell)
      elif cell in set2:
        c.union(side2, cell)
      c.union(i, cell)
    self.miai_connections[ptm] = c

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
    self.move(ch, where)
    return True

  def move(self, ch, where):
    self.H.append((self.brd[where], where, self.connection_graphs, deepcopy(self.miai_reply), deepcopy(self.miai_connections)))
    self.brd = change_str(self.brd, where, ch)
    if ch != ECH:
      self.update_miai_at(self.miai_reply, where)
      self.update_miai_connections(ch, where)
    else:
      self.miai_reply = self.get_miai_replies()
      self.miai_connections = {BCH:self.get_miai_connections(BCH), WCH:self.get_miai_connections(WCH)}
    self.connection_graphs = {BCH:self.get_connections(BCH), WCH:self.get_connections(WCH)}

  def has_win(self, ptm):
    connections, side1, side2 = self.connection_graphs[ptm]
    # Check if the special side nodes are adjacent in the connection graph
    return side1 in connections[side2]

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

  def live_cells(self, ptm):
    connections, side1, side2 = self.connection_graphs[ptm]
    paths = self.induced_paths_from_to(connections, set(), side1, side2)
    live = set()
    for p in paths:
      for pt in p:
        live.add(pt)
    if side1 in live:
      live.remove(side1)
    if side2 in live:
      live.remove(side2)
    return live

  def induced_paths_from_to(self, connections, visited, node, end):
    if node == end:
      return {(end,)}
    visited.add(node)
    paths = set()
    candidates = set()
    for n in connections[node]:
      # If the function was called on this node it must be a candidate
      # So the only node that could be in visited that is connected to it
      # would be the previous node. So don't go back to that one.
      if n in visited:
        continue
      is_candidate = True
      for n1 in connections[n]:
        if n1 in visited and n1 != node:
          is_candidate = False
          break
      if is_candidate:
        candidates.add(n)

    if not candidates:
      visited.remove(node)
      # Path does not end at end so discard it
      return {}
    for n in candidates:
      p1 = self.induced_paths_from_to(connections, visited, n, end)
      for p in p1:
        paths.add((node,) + p)
    visited.remove(node)
    return paths

  def shortest_paths_from_to(self, connections, node, end):
    # Find the nodes contained in all shortest paths between two nodes using modified bfs
    # Don't need to find the exact paths, just backtrack from end to start using parents
    parents = [[] for i in range(len(self.brd) + 2)]
    dists = [math.inf for i in range(len(self.brd)+2)]
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
          parents[n1] = [n]
          q.append(n1)
        elif dists[n1] == d:
          parents[n1].append(n)

    # Find all nodes in the shortest paths, add them to seen
    seen = {}
    q = deque([[end]])
    while q:
      p = q.popleft()
      for v in p:
        if v not in seen:
          seen[v] = 1
        else:
          seen[v] += 1
        q.append(parents[v])

    seen.pop(node)
    seen.pop(end)
    #counts = sorted([(seen[key], key) for key in seen.keys()])
    #return [k[1] for k in counts]
    return seen.keys()

  def rank_moves_by_vc(self, ptm, show_ranks=False):
    # Assign a score to each node based on whether it is virtually connected to other nodes/sides and
    # on whether it is in a shortest winning path
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    optm = oppCH(ptm)
    score = [0] * len(self.brd)
    for i in range(len(self.brd)):
      if self.brd[i] != ECH:
        continue

      # Find possible vcs
      poss_vcs = set()
      for c in self.nbrs[i]:
        if self.brd[c] != ECH:
          continue
        for c1 in self.nbrs[c]:
          if self.brd[c1] != ECH:
            continue
          elif c1 == i: continue
          if c1 in self.nbrs[i]:
            poss_vcs.add(tuple(sorted((c, c1))))

      # Check each possible vc to see if it is an actual vc
      for pvc in poss_vcs:
        # Add to score if it could be a vc in the future
        if self.brd[pvc[0]] == ECH and self.brd[pvc[1]] == ECH:
          score[i] += 1

        # Add to score if it is an actual vc or vc'd to the edge
        s = set(self.nbrs[pvc[0]]).intersection(set(self.nbrs[pvc[1]]))
        s.remove(i)
        if not s:
          if (pvc[0] in set1 and pvc[1] in set1) or (pvc[0] in set2 and pvc[1] in set2):
            score[i] += 1
        elif self.brd[list(s)[0]] == ptm:
          score[i] += 1

    spft = self.shortest_paths_from_to(*self.connection_graphs[ptm])
    # 5 is an arbitrary constant that seemed to work well
    for i in spft:
      score[i] += 5
    counts = sorted([(score[i], i) for i in range(len(self.brd))], reverse=True)
    if show_ranks:
      print(counts)
    return [i[1] for i in counts]

  def midpoint(self):
    x = self.C // 2
    y = self.R // 2 + self.R % 2 - 1
    return coord_to_point(y, x, self.C)

  def captured(self, ptm):
    # Uses patterns to find cells captured by the current player.
    # Does not find all captured cells.
    inf_cs = set()
    #for i in range(len(self.brd)):
      #for pat in self.dc_patterns:
        #inf_cs = inf_cs.union(pat.matches(self.brd, i))
    if ptm == WCH:
      for i in range(len(self.brd)):
        for pat in self.wc_patterns:
          inf_cs = inf_cs.union(pat.matches(self.brd, i))
    elif ptm == BCH:
      for i in range(len(self.brd)):
        for pat in self.bc_patterns:
          inf_cs = inf_cs.union(pat.matches(self.brd, i))
    return inf_cs

  def win_move(self, ptm): # assume neither player has won yet
    #self.showboard()
    optm = oppCH(ptm) 
    calls = 1
    ovc = set()
    #cap = self.captured(ptm)

    mustplay = set([i for i in range(len(self.brd)) if self.brd[i] == ECH])
    #mustplay = self.live_cells(ptm)
    while len(mustplay) > 0:
      cells = self.rank_moves_by_vc(ptm) # self.CELLS
      if self.H:
        miai_replies = self.miai_reply[ptm][self.H[-1][1]]
        cells = list(miai_replies) + cells
      else:
        cells = [self.midpoint()] + cells
      # Find first empty cell
      for move in cells:
        if move in mustplay: break

      self.move(ptm, move)
      #self.showboard()
      #input()
      
      if self.has_win(ptm):
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

  def undo(self):  # pop last meta-move
    if not self.H:
      print('\n    original position,  nothing to undo\n')
      return False
    else:
      ch, where, self.connection_graphs, self.miai_reply, self.miai_connections = self.H.pop()
      self.brd = change_str(self.brd, where, ch)
    return True

  def msg(self, ch):
    if self.has_win('x'): return('x has won')
    elif self.has_win('o'): return('o has won')
    else:
      st = time.time()
      wm, calls, vc = self.win_move(ch)
      out = '\n' + ch + '-to-move: '
      out += (ch if wm else oppCH(ch)) + ' wins' 
      out += (' ... ' if wm else ' ') + wm + '\n'
      out += str(calls) + ' calls\n'
      out += "%.4f" % (time.time() - st) + 's \n'
      return out


def printmenu():
  print('  h                              help menu')
  print('  ? x|o      solve the position for x or o')
  print('  z                         show the board')
  print('  x b2                          play x b 2')
  print('  o e3                          play o e 3')
  print('  . a2                           erase a 2')
  print('  u                                   undo')
  print('  [return]                            quit')


def interact():
  p = Position(4, 4)
  while True:
    p.showboard()
    cmd = input(' ').split()
    if len(cmd)==0:
      print('\n ... adios :)\n')
      return
    if cmd[0]=='h':
      printmenu()
    elif cmd[0]=='z':
      try:
        sz = int(cmd[1])
        if (sz > 0):
          p = Position(sz, sz)
      except:
        pass
    elif cmd[0]=='u':
      if len(cmd) > 1:
        for i in range(int(cmd[1])):
          if not p.undo(): break
      else:
        p.undo()
    elif cmd[0]=='?':
      if len(cmd)>0:
        if cmd[1]=='x': 
          print(p.msg('x'))
        elif cmd[1]=='o': 
          print(p.msg('o'))
    #elif cmd[0] == 'l':
      #print(" ".join(sorted([point_to_alphanum(x, p.C) for x in p.live_cells(cmd[1])])))
    #elif cmd[0] == "rm":
      #p.rank_moves_by_vc(cmd[1], show_ranks=True)
    elif cmd[0] == "mc":
      ch = cmd[1]
      c = p.get_miai_connections(ch)
      cg, side1, side2 = p.connection_graphs[ch]
      print(c.find(side1)==c.find(side2))
      pass
    elif cmd[0] == "c":
      print(p.captured(cmd[1]))
    elif (cmd[0] in PTS):
      p.requestmove(cmd[0] + ' ' + ''.join(cmd[1:]))


#if __name__ == "__main__":
interact()
