"""Copyright [2017] [Siddhant Mahapatra]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/Robosid/Artificial-Intelligence/blob/master/License.pdf
    https://github.com/Robosid/Artificial-Intelligence/blob/master/License.rtf

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import two_test_a_star as pf

import unittest


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_maze(self):
        a = pf.AStar()
        walls = ((0, 5), (1, 0), (1, 1), (1, 5), (2, 3),
                 (3, 1), (3, 2), (3, 5), (4, 1), (4, 4), (5, 1))
        a.init_grid(6, 6, walls, (0, 0), (5, 5))
        path = a.solve()
        self.assertEqual(path, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4),
                                (2, 4), (3, 4), (3, 3), (4, 3), (5, 3), (5, 4),
                                (5, 5)])

    def test_maze_no_walls(self):
        a = pf.AStar()
        walls = ()
        a.init_grid(6, 6, walls, (0, 0), (5, 5))
        path = a.solve()
        self.assertEqual(len(path), 11)

    def test_maze_no_solution(self):
        a = pf.AStar()
        walls = ((0, 5), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                 (2, 3), (3, 1), (3, 2), (3, 5), (4, 1), (4, 4), (5, 1))
        a.init_grid(6, 6, walls, (0, 0), (5, 5))
        self.assertIsNone(a.solve())

if __name__ == '__main__':
unittest.main()