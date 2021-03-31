"""
negamax small-board hex solver
"""
from copy import deepcopy
import time
import math
from collections import deque
import numpy as np

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

def alphanum_to_point(an, C):
  return (ord(an[0]) - ord('a')) + (int(an[1:])-1)*C

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

  def __str__(self):
    return str(self.parents)


class Pattern:
  # Represents a pattern of cells to match to cells on the board, could be a captured pattern or a dead cell pattern
  def __init__(self, offsets, chars, rows, cols):
    # Offsets are vectors representing offsets from the main cell of the pattern at [0, 0, 0]
    # Main cell should be an empty cell so that it is always on the board and can potentially match more places.
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
      self.deltas.append([cubic_rotate_60_cc(np.array(v)) for v in prev])
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
    return ch != ECH and (x == self.C and z == -1) or (x == -1 and z == self.R)


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
    self.miai_connections, self.miai_reply, self.ws, self.vcs = self.vcs_bp()
    self.voltage = {BCH:self.compute_voltage(BCH), WCH:self.compute_voltage(WCH)}

    # dead cell patterns
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
    # black captured patterns
    self.bc_patterns = [
      # x x x
      #  * *
      #   x
      Pattern([[0, 0, 0], [1, -1, 0], [0, -1, 1], [0, 1, -1], [1, 0, -1], [2, -1, -1]],
              [ECH, ECH, BCH, BCH, BCH, BCH], self.R, self.C),
      # x x
      #  * * o
      #   x
      Pattern([[0, 0, 0], [1, -1, 0], [0, 1, -1], [1, 0, -1], [2, -2, 0], [0, -1, 1]],
              [ECH, ECH, BCH, BCH, WCH, BCH], self.R, self.C),
      Pattern([[0, 0, 0], [1, -1, 0], [1, 0, -1], [2, -1, -1], [-1, 1, 0], [0, -1, 1]],
              [ECH, ECH, BCH, BCH, WCH, BCH], self.R, self.C),
      #   x x
      #  * *
      # x x
      Pattern([[1, 0, -1], [2, -1, -1], [0, 0, 0], [1, -1, 0], [-1, 0, 1], [0, -1, 1]],
              [BCH, BCH, ECH, ECH, BCH, BCH], self.R, self.C),
      Pattern([[0, 1, -1], [1, 0, -1], [0, 0, 0], [1, -1, 0], [0, -1, 1], [1, -2, 1]],
              [BCH, BCH, ECH, ECH, BCH, BCH], self.R, self.C),
      # x * * x
      #  x x x
      Pattern([[-1, 1, 0], [0, 0, 0], [1, -1, 0], [2, -2, 0], [-1, 0, 1], [0, -1, 1], [1, -2, 1]],
              [BCH, ECH, ECH, BCH, BCH, BCH, BCH], self.R, self.C),
      # o
      #  * * x
      # x x x
      Pattern([[0, 1, -1], [0, 0, 0], [1, -1, 0], [2, -2, 0], [-1, 0, 1], [0, -1, 1], [1, -2, 1]],
              [WCH, ECH, ECH, BCH, BCH, BCH, BCH], self.R, self.C),
      Pattern([[1, 0, -1], [0, 0, 0], [-1, 1, 0], [-2, 2, 0], [0, -1, 1], [-1, 0, 1], [-2, 1, 1]],
              [WCH, ECH, ECH, BCH, BCH, BCH, BCH], self.R, self.C),
    ]

    # white captured patterns
    self.wc_patterns = [
      # o o o
      #  * *
      #   o
      Pattern([[0, 0, 0], [1, -1, 0], [0, -1, 1], [0, 1, -1], [1, 0, -1], [2, -1, -1]],
              [ECH, ECH, WCH, WCH, WCH, WCH], self.R, self.C),
      # o o
      #  * * x
      #   o
      Pattern([[0, 0, 0], [1, -1, 0], [0, 1, -1], [1, 0, -1], [2, -2, 0], [0, -1, 1]],
              [ECH, ECH, WCH, WCH, BCH, WCH], self.R, self.C),
      Pattern([[0, 0, 0], [1, -1, 0], [1, 0, -1], [2, -1, -1], [-1, 1, 0], [0, -1, 1]],
              [ECH, ECH, WCH, WCH, BCH, WCH], self.R, self.C),
      #   o o
      #  * *
      # o o
      Pattern([[1, 0, -1], [2, -1, -1], [0, 0, 0], [1, -1, 0], [-1, 0, 1], [0, -1, 1]],
              [WCH, WCH, ECH, ECH, WCH, WCH], self.R, self.C),
      Pattern([[0, 1, -1], [1, 0, -1], [0, 0, 0], [1, -1, 0], [0, -1, 1], [1, -2, 1]],
              [WCH, WCH, ECH, ECH, WCH, WCH], self.R, self.C),
      # o * * o
      #  o o o
      Pattern([[-1, 1, 0], [0, 0, 0], [1, -1, 0], [2, -2, 0], [-1, 0, 1], [0, -1, 1], [1, -2, 1]],
              [WCH, ECH, ECH, WCH, WCH, WCH, WCH], self.R, self.C),
      # x
      #  * * o
      # o o o
      Pattern([[0, 1, -1], [0, 0, 0], [1, -1, 0], [2, -2, 0], [-1, 0, 1], [0, -1, 1], [1, -2, 1]],
              [BCH, ECH, ECH, WCH, WCH, WCH, WCH], self.R, self.C),
      Pattern([[1, 0, -1], [0, 0, 0], [-1, 1, 0], [-2, 2, 0], [0, -1, 1], [-1, 0, 1], [-2, 1, 1]],
              [BCH, ECH, ECH, WCH, WCH, WCH, WCH], self.R, self.C),
    ]


  def compute_voltage(self, ptm, voltages=None, max_delta=0.00001):
    #WARNING Doesn't work yet
    optm = oppCH(ptm)
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    g, side1, side2 = self.connection_graphs[ptm]
    # Voltage flows from side1 to side2
    if not voltages:
      voltages = [0.0] * (self.R * self.C) + [1.0, 0.0]
    err = max_delta
    keys = g.keys() - {side1, side2} # Don't update source or sink
    occ = {i for i in range(len(self.brd)) if self.brd[i] == ptm}
    empty = keys - occ
    # TODO: Better ordering for iteratively computing voltages?
    while err >= max_delta:
      err = 0.0
      for node in occ:
        nbrs = self.nbrs[node]
        if node in set1:
          nbrs = nbrs | {side1}
        if node in set2:
          nbrs = nbrs | {side2}
        v = 0.0
        for nbr in nbrs:
          v = max(v, voltages[nbr])
        err = max(err, v - voltages[node])
        voltages[node] = v
      for node in empty:
        nbrs = self.nbrs[node]
        if node in set1:
          nbrs = nbrs | {side1}
        if node in set2:
          nbrs = nbrs | {side2}
        v = 0.0
        for nbr in nbrs:
          v += voltages[nbr]
        v /= len(nbrs)
        err = max(err, v - voltages[node])
        voltages[node] = v
    return voltages

  def voltage_drops(self, ptm):
    #WARNING Doesn't work yet
    g, side1, side2 = self.connection_graphs[ptm]
    #set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    optm = oppCH(ptm)
    voltage = self.voltage[ptm]
    vdrops = []
    for i in range(len(self.brd)):
      if self.brd[i] != ECH:
        continue
      nbrs = g[i]
      drop = 0.0
      for nbr in nbrs:
        if nbr < len(self.brd) and self.brd[nbr] == optm:
          continue
        drop += max(0, voltage[nbr] - voltage[i])
      vdrops.append((drop, i))
    vdrops = [pair[1] for pair in sorted(vdrops, reverse=True)]
    return vdrops

  def update_voltage(self, ptm, move, max_delta=0.0001):
    optm = oppCH(ptm)
    vp = self.voltage[ptm]
    vo = self.voltage[optm]
    vo[move] = 0.0
    return {ptm:self.compute_voltage(ptm, voltages=vp, max_delta=max_delta), optm:self.compute_voltage(optm, voltages=vo, max_delta=max_delta)}

  def vc_search(self, ptm):
    # Search for virtual connections
    # Does not find all virtual connections but can detect 432s.

    def sort2(l):
      if l[0] < l[1]:
        return l
      return (l[1], l[0])

    def construct_miai(conn, miai_build, miai_reply):
      # Find miai by building virtual connections from pairs of semiconnections
      if conn not in miai_build:
        return
      m = miai_build[conn]
      for c in m:
        construct_miai(c, miai_build, miai_reply)
      scs = [i for i in m if len(i) == 3]
      subs = {(sc, sc1) for sc in scs for sc1 in scs if sc != sc1}
      for pair in subs:
        b1 = pair[0][2]
        b2 = pair[1][2]
        miai_reply[b1].add(b2)
        miai_reply[b2].add(b1)

    optm = oppCH(ptm)
    _, side1, side2 = self.connection_graphs[ptm]
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    cells = {i for i in range(len(self.brd)) if self.brd[i] != optm}
    c_n_s = cells | {side1, side2}
    miai_conn = UnionFind()
    miai_reply = [set() for i in range(len(self.brd))]
    scs = {}
    vcs = {}
    miai_build = {}

    # Initialize vcs
    for i in range(len(self.brd)):
      if self.brd[i] == optm:
        continue
      for j in self.nbrs[i]:
        if self.brd[j] == optm:
          continue
        s = sort2((i, j))
        if s not in vcs:
          vcs[s] = set()
        if self.brd[i] == ptm and self.brd[j] == ptm:
          miai_conn.union(i, j)
    for loc in set1:
      ch = self.brd[loc]
      if ch == optm:
        continue
      elif ch == ptm:
        miai_conn.union(side1, loc)
      vcs[(loc, side1)] = set()
    for loc in set2:
      ch = self.brd[loc]
      if ch == optm:
        continue
      elif ch == ptm:
        miai_conn.union(side2, loc)
      vcs[(loc, side2)] = set()

    loop = True
    while loop:
      loop = False
      # Find new vcs/scs from vcs
      for b in cells:
        # Find vcs that both end at b
        enb = {vc for vc in vcs.keys() if b in vc}
        for vc in enb:
          for vc1 in enb - {vc}:
            # Get the ends of each vc
            a = next(iter(set(vc)-{b}))
            c = next(iter(set(vc1)-{b}))
            s = sort2((a, c))
            k = s + (b,)
            if s in vcs:
              continue
            if k in scs:
              continue
            # Found a new vc or sc
            loop = True
            if self.brd[b] == ptm:
              # Found a virtual connection
              # Carrier for this vc is the union of the carriers for vc and vc1
              vcs[s] = vcs[vc] | vcs[vc1]
              # If both sides are occupied with a ptm stone update connectivity
              if (s[0] in {side1, side2} or self.brd[s[0]] == ptm) and (s[1] in {side1, side2} or self.brd[s[1]] == ptm):
                miai_conn.union(a, c)
              miai_build[s] = (vc, vc1)
            else:
              # Found a semiconnection
              # b is the key, the carrier is the union of the carriers and also includes b
              scs[k] = vcs[vc] | vcs[vc1] | {b}
              miai_build[k] = (vc, vc1)

      # Find new vcs from scs
      for a in c_n_s:
        for c in c_n_s:
          s = sort2((a, c))
          if s in vcs:
            continue
          # Get all semiconnections with common ends
          enb = {sc for sc in scs.keys() if sc[:2] == s}
          subs = {(sc, sc1) for sc in enb for sc1 in enb if sc != sc1}
          for pair in subs:
            sc1 = pair[0]
            sc2 = pair[1]
            set1 = scs[sc1]
            set2 = scs[sc2]
            # If the intersection of their carriers is empty then they form a new vc
            if not set1 & set2:
              vcs[s] = set1 | set2
              miai_build[s] = (sc1, sc2)
              if (s[0] in {side1, side2} or self.brd[s[0]] == ptm) and (s[1] in {side1, side2} or self.brd[s[1]] == ptm):
                miai_conn.union(a, c)
                construct_miai(s, miai_build, miai_reply)
              loop = True
              break

    ws = set()
    if (side1, side2) in vcs:
      ws = vcs[(side1, side2)]
    return miai_conn, miai_reply, ws, vcs

  def vcs_bp(self):
    # Get miai info for both players
    mcb, mrb, wsb, vcsb = self.vc_search(BCH)
    mcw, mrw, wsw, vcsw = self.vc_search(WCH)
    # return miai connections, then miai responses, and a winset if both sides are vc'd otherwise the empty set. Also vcs
    return {BCH:mcb, WCH:mcw}, {BCH:mrb, WCH:mrw}, {BCH:wsb, WCH:wsw}, {BCH:vcsb, WCH:vcsw}

  #TODO: If we have a reply to an opponent probe that restores a connection, should we record that information?

  def miai_connected(self, ptm):
    # Check efficiently whether ptm stones are miai connected
    c = self.miai_connections[ptm]
    conn, side1, side2 = self.connection_graphs[ptm]
    return c.find(side1) == c.find(side2)

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
    self.H.append((self.brd[where], where, self.miai_reply, self.miai_connections, self.ws, self.connection_graphs, deepcopy(self.voltage), deepcopy(self.vcs)))
    self.brd = change_str(self.brd, where, ch)
    self.connection_graphs = {BCH:self.get_connections(BCH), WCH:self.get_connections(WCH)}
    self.miai_connections, self.miai_reply, self.ws, self.vcs = self.vcs_bp()
    self.voltage = self.update_voltage(ch, where) #{BCH:self.compute_voltage(BCH), WCH:self.compute_voltage(WCH)}

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
    # Warning!!! Slow for larger boards!
    connections, side1, side2 = self.get_connections(ptm)
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
    # Warning!!! Slow for larger boards!
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

    if node in seen:
      seen.pop(node)
    if end in seen:
      seen.pop(end)
    #counts = sorted([(seen[key], key) for key in seen.keys()])
    #return [k[1] for k in counts]
    return seen.keys()

  def rank_moves_by_vc(self, ptm, show_ranks=False, recurse=True):
    # Assign a score to each node based on whether it is virtually connected to other nodes/sides
    # and/or whether it is in a shortest winning path
    set1, set2 = (self.TOP_ROW, self.BTM_ROW) if ptm == BCH else (self.LFT_COL, self.RGT_COL)
    _, side1, side2 = self.connection_graphs[ptm]
    optm = oppCH(ptm)
    score = [0] * len(self.brd)

    if recurse:
      opp_rnk = self.rank_moves_by_vc(optm, recurse=False)
      for i in range(len(opp_rnk))[:5]:
        score[opp_rnk[i]] += 5 #(len(opp_rnk)-i)/len(opp_rnk)

    if self.H:
      miai_replies = self.miai_reply[ptm][self.H[-1][1]]
      for mr in miai_replies:
        score[mr] += 12

    # Add to score of a position if it is is contained in a vc
    vcs = self.vcs[ptm]
    for i in range(len(self.brd)):
      if self.brd[i] != ECH:
        continue

      for vc in vcs:
        if vc[0] == i:
          score[vc[0]] += 1
          if vc[1] not in {side1, side2} and self.brd[vc[1]] == ECH:
            score[vc[0]] += 1
            score[vc[1]] += 1
        elif vc[1] == i:
          score[vc[1]] += 1
          if vc[0] not in {side1, side2} and self.brd[vc[0]] == ECH:
            score[vc[0]] += 1
            score[vc[1]] += 1

    #spft = self.shortest_paths_from_to(*self.connection_graphs[ptm])
    # Scores are arbitrary constants that seemed to work well
    #for i in spft:
    #  score[i] += 5
    for i in self.voltage_drops(ptm)[:5]:
      score[i] += 5
    counts = sorted([(score[i], i) for i in range(len(self.brd))], reverse=True)
    if show_ranks:
      print("Cells:Ranks", " ".join([point_to_alphanum(rc[1], self.C) + ":" + str(rc[0]) for rc in counts])) 
    return [i[1] for i in counts]

  def midpoint(self):
    x = self.C // 2
    y = self.R // 2 + self.R % 2 - 1
    return coord_to_point(y, x, self.C)

  def dead(self):
    # Uses patterns to find dead cells.
    # Does not find all dead cells.
    inf_cs = set()
    for i in range(len(self.brd)):
      if self.brd[i] != ECH:
        continue
      for pat in self.dc_patterns:
        inf_cs = inf_cs.union(pat.matches(self.brd, i))
    return inf_cs

  def captured(self, ptm):
    # Uses patterns to find cells captured by the current player.
    # Does not find all captured cells.
    inf_cs = set()
    c_pats = self.bc_patterns
    if ptm == WCH:
      c_pats = self.wc_patterns
    for i in range(len(self.brd)):
      if self.brd[i] != ECH:
        continue
      for pat in c_pats:
        inf_cs = inf_cs.union(pat.matches(self.brd, i))
    return inf_cs

  def fill_cells(self, cells, ptm):
    i = 0
    for c in cells:
      if self.brd[c] == ECH:
        self.move(ptm, c)
        i += 1
    return i

  def refresh(self):
    self.miai_connections, self.miai_reply, self.ws, self.vcs = self.vcs_bp()

  def fill_cells_lite(self, cells, ptm):
    i = 0
    for c in cells:
      if self.brd[c] == ECH:
        self.brd = change_str(self.brd, c, ptm)
        i += 1
    return i

  def win_move(self, ptm, captured={BCH:set(), WCH:set()}):
    # assume neither player has won yet
    optm = oppCH(ptm) 
    calls = 1
    ovc = set()

    if self.miai_connected(ptm):
        ws = self.ws[ptm] | captured[ptm]
        return point_to_alphanum(next(iter(ws)), self.C), calls, ws

    if self.miai_connected(optm):
      return '', calls, set()

    mustplay = {i for i in range(len(self.brd)) if self.brd[i] == ECH}
    cells = self.rank_moves_by_vc(ptm)
    #if not self.H:
    #  cells = [self.midpoint()] + cells

    while len(mustplay) > 0:
      move = None
      for i in range(len(cells)):
        mv = cells[i]
        if mv in mustplay:
          move = mv
          cells = cells[i+1:]
          break

      brd = self.brd
      self.brd = change_str(self.brd, move, ptm)

      pcap = captured[ptm]
      ocap = captured[optm]
      while True:
        d = self.dead()
        self.fill_cells_lite(d, ptm)
        cp = self.captured(ptm)
        pcap = pcap | cp
        self.fill_cells_lite(cp, ptm)
        co = self.captured(optm)
        ocap = ocap | co
        self.fill_cells_lite(co, optm)
        if not (d or cp or co):
          break
      self.refresh()

      if self.miai_connected(ptm): # Also true if has_win
        pt = point_to_alphanum(move, self.C)
        ws = self.ws[ptm] | pcap
        self.brd = brd
        self.refresh()
        return pt, calls, ws.union({move})

      omv, ocalls, oset = self.win_move(optm, {ptm:pcap, optm:ocap})

      calls += ocalls
      if not omv: # opponent has no winning response to ptm move
        oset.add(move)
        pt = point_to_alphanum(move, self.C)
        self.brd = brd
        self.refresh()
        return pt, calls, oset

      ovc = ovc.union(oset)
      mustplay = mustplay.intersection(oset)
      self.brd = brd
      self.refresh()

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

  def show_miai_info(self, ch):
    sides = {BCH:["T", "B"],WCH:["L", "R"]}
    miai_colour = escape_ch + '0;31m'
    c = self.miai_connections[ch]
    conn, side1, side2 = self.connection_graphs[ch]
    info = [ECH] * (self.R * self.C + 2)
    colours = [None for i in info]
    for i in range(len(self.brd)):
      if self.brd[i] == ch:
        pos = c.find(i)
        if pos >= len(self.brd):
          info[i] = sides[ch][pos % len(self.brd)]
        else:
          info[i] = point_to_alphanum(pos, self.C)
        colours[i] = miai_colour
      elif self.brd[i] == ECH:
        mr = [r for r in self.miai_reply[ch][i] if self.brd[r] == ECH]
        if mr:
          info[i] = point_to_alphanum(mr[0], self.C)
    self.show_big_board(info, colours)
    print("Miai connected:", c.find(side1)==c.find(side2))

  def undo(self):  # pop last meta-move
    if not self.H:
      print('\n    original position,  nothing to undo\n')
      return False
    else:
      ch, where, self.miai_reply, self.miai_connections, self.ws, self.connection_graphs, self.voltage, self.vcs = self.H.pop()
      self.brd = change_str(self.brd, where, ch)
    return True

  def msg(self, ch):
    if self.miai_connected('x'): return('x has won')
    elif self.miai_connected('o'): return('o has won')
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
  print('  m                      display miai info')
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

    #elif cmd[0] == 'l':
      #print(" ".join(sorted([point_to_alphanum(x, p.C) for x in p.live_cells(cmd[1])])))

    elif cmd[0] == "rm":
      if len(cmd) != 2:
        print("Command requires one argument.")
        continue
      if cmd[1] not in {BCH, WCH}:
        print("Argument must be one of", BCH, WCH)
        continue
      print()
      p.rank_moves_by_vc(cmd[1], show_ranks=True)

    #elif cmd[0] == "mws":
      #print(p.get_miai_ws(cmd[1]))

    elif cmd[0] == "m":
      if len(cmd) == 2 and cmd[1] in {WCH, BCH}:
        p.show_miai_info(cmd[1])

    elif cmd[0] == "c":
      dead = set()
      capb = set()
      capw = set()
      brd = p.brd
      while True:
        d = p.dead()
        dead = dead.union(d)
        cb = p.captured(BCH)
        capb = capb.union(cb)
        cw = p.captured(WCH)
        capw = capw.union(cw)
        p.fill_cells_lite(d, BCH)
        p.fill_cells_lite(cb, BCH)
        p.fill_cells_lite(cw, WCH)
        if not (d or cb or cw):
          break
      print()
      print("Dead:", " ".join([point_to_alphanum(x, p.C) for x in dead]))
      print("BCap:", " ".join([point_to_alphanum(x, p.C) for x in capb]))
      print("WCap:", " ".join([point_to_alphanum(x, p.C) for x in capw]))
      p.showboard()
      p.brd = brd

    elif (cmd[0] in PTS):
      p.requestmove(cmd[0] + ' ' + ''.join(cmd[1:]))

    elif (cmd[0] == 's'):
      p.showboard()

    elif (cmd[0] == 'v'):
      try:
        v = p.compute_voltage(cmd[1])
        for i in range(len(p.brd)):
          print(point_to_alphanum(i, p.C), v[i])
        print([point_to_alphanum(i, p.C) for i in p.voltage_drops(cmd[1])])
      except:
        print("Please supply a valid player to compute voltages for.")

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
