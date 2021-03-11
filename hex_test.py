import unittest
from hex_simple import Position, BCH, ECH, WCH

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
        self.assertEqual(pos.get_all_miai_ws(WCH), {4, 9, 7, 12, 10, 15})


    def test_pattern(self):
        pos = Position(10,10)
        pos.fill_cells([3, 5], BCH)
        pos.fill_cells([16, 18, 26, 27], BCH)
        pos.fill_cells([14, 24, 34], WCH)
        pos.fill_cells([80, 81, 91], BCH)
        self.assertEqual(pos.dead(), {4, 17, 25, 90})
        self.assertEqual(pos.captured(BCH), {6, 7, 8, 9, 15})


    def test_correct(self):
        pos = Position(4,4)
        pos.move(BCH, 3)
        self.assertFalse(pos.win_move(WCH)[0])
        pos.undo()
        pos.move(BCH, 6)
        self.assertFalse(pos.win_move(WCH)[0])

        pos = Position(5,5)
        for i in range(25):
            pos.move(BCH, i)
            print("Moved to", i)
            if i in {0, 1, 2, 3, 5, 10, 24, 23, 22, 21, 19, 14}:
                self.assertTrue(pos.win_move(WCH)[0])
            else:
                self.assertFalse(pos.win_move(WCH)[0])
            pos.undo()
                
                    

if __name__ == '__main__':
    unittest.main()
