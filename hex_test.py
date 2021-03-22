import unittest
import time
from hex_simple import Position, BCH, ECH, WCH, PTS, alphanum_to_point
import random
import os
from sgf_parse import SgfTree

class TestHex(unittest.TestCase):
    def test_miai(self):
        pos = Position(5,5)
        pos.move(BCH, 3)
        self.assertFalse(pos.miai_connected(BCH))
        pos.move(BCH, 12)
        self.assertFalse(pos.miai_connected(BCH))
        pos.move(BCH, 21)
        self.assertTrue(pos.miai_connected(BCH))
        self.assertEqual({7}, pos.miai_reply[BCH][8])
        self.assertEqual({8}, pos.miai_reply[BCH][7])
        pos.move(WCH, 8)
        self.assertFalse(pos.miai_connected(BCH))
        pos.move(WCH, 11)
        self.assertFalse(pos.miai_connected(BCH))
        self.assertFalse(pos.miai_connected(WCH))
        pos.move(ECH, 12)
        self.assertFalse(pos.miai_connected(BCH))
        self.assertTrue(pos.miai_connected(WCH))
        self.assertEqual(pos.get_all_miai_ws(WCH), {4, 9, 7, 8, 11, 12, 10, 15})


    def test_pattern(self):
        pos = Position(10,10)
        pos.fill_cells([3, 5], BCH)
        pos.fill_cells([16, 18, 26, 27], BCH)
        pos.fill_cells([14, 24, 34], WCH)
        pos.fill_cells([80, 81, 91], BCH)
        self.assertEqual(pos.dead(), {4, 17, 25, 90})
        self.assertEqual(pos.captured(BCH), {6, 7, 8, 9, 15})


    @unittest.skip("Takes a long time")
    def test_correct(self):
        pos = Position(5,5)
        for i in range(25):
            pos.move(BCH, i)
            st = time.time()
            print("Moved to", i)
            if i in {0, 1, 2, 3, 5, 10, 24, 23, 22, 21, 19, 14}:
                self.assertTrue(pos.win_move(WCH, set())[0])
            else:
                self.assertFalse(pos.win_move(WCH, set())[0])
            pos.undo()
            print("%.4f" % (time.time() - st) + 's \n')


    def test_sgf(self):
        for i in range(1024):
            rows = random.randint(2, 8)
            cols = random.randint(2, 8)
            p = Position(rows, cols)
            for i in range(len(p.brd)):
                p.move(random.choice(PTS), i)
            brd = p.brd
            fname = "./temporary_test_file_do_not_use_this_name.sgf"
            if os.path.exists(fname):
                raise Exception("Test file already exists")
            p.save(fname)
            f = open(fname, 'r')           
            t = None
            t = SgfTree(f.read())
            f.close()
            os.remove(fname)
            sz = t.children[0].properties["SZ"][0].split(':')
            if len(sz) == 1:
                r = int(sz[0])
                p = Position(r, r)
            elif len(sz) == 2:
                p = Position(int(sz[1]), int(sz[0]))
            else:
                raise Exception("Bad")

            self.assertEqual(p.R, rows)
            self.assertEqual(p.C, cols)
            while t.children:
                t1 = t.children[0]
                props = t1.properties
                chrs = {"B":BCH, "W":WCH}
                for ch in chrs:
                    if ch in props:
                        p.move(chrs[ch], alphanum_to_point(props[ch][0], p.C))
                t = t1
            self.assertEqual(p.brd, brd)


if __name__ == '__main__':
    unittest.main()
