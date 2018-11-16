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

from __future__ import print_function
from __future__ import division


from sokoban import Warehouse

from mySokobanSolver import my_team, taboo_cells, SokobanPuzzle, check_action_seq
from mySokobanSolver import solve_sokoban_elem, can_go_there, solve_sokoban_macro 

puzzle_t1 ='''
#######
#@ $. #
#######'''

puzzle_t2 ='''
  #######
  #     #
  # .$. #
 ## $@$ #
 #  .$. #
 #      #
 ########
'''

puzzle_t3 ='''
#######
#@ $ .#
#. $  #
#######'''

expected_answer_3 ='''
#######
#X    #
#    X#
#######'''


expected_answer_1 =''' 
 ####
 # .#
 #  ###
 #*   #
 #  $@#
 #  ###
 ####
'''


def test_warehouse_1():
    wh = Warehouse()
    # read the puzzle from the multiline string 
    wh.extract_locations(puzzle_t2.split('\n'))
    print("\nPuzzle from multiline string")
    print(wh)

def test_warehouse_2():
    problem_file = "./warehouses/warehouse_01.txt"
    wh = Warehouse()
    wh.read_warehouse_file(problem_file)
    print("\nPuzzle from file")
    print(wh)
    print(wh.worker) # x,y  coords !!
    print(wh.walls)  # x,y  coords !!
    
def test_taboo_cells():
    wh = Warehouse()
    wh.extract_locations(puzzle_t3.split('\n'))
    answer = taboo_cells(wh)
    assert( answer == expected_answer_3 )


def test_check_elem_action_seq():
    problem_file = "./warehouses/warehouse_01.txt"
    wh = Warehouse()
    wh.read_warehouse_file(problem_file)
    answer = check_action_seq(wh, ['Right', 'Right','Down'])
    assert( answer == expected_answer_1)

def test_solve_sokoban_elem():
    problem_file = "./warehouses/warehouse_01.txt"
    wh = Warehouse()
    wh.read_warehouse_file(problem_file)
    answer = solve_sokoban_elem(wh)
    assert( answer ==  ['Right', 'Right'])

def test_can_go_there():
    problem_file = "./warehouses/warehouse_01.txt"
    wh = Warehouse()
    wh.read_warehouse_file(problem_file)
    answer = can_go_there(wh,(30,2))
    assert( answer ==  False)
    answer = can_go_there(wh,(6,2))
    assert( answer ==  True)
    
  
def test_solve_sokoban_macro():
    wh = Warehouse()
    wh.extract_locations(puzzle_t3.split('\n'))
    print(wh)
    answer = solve_sokoban_macro(wh)
    assert( answer ==  [ ((2,3),'Right'), ((2,4),'Right'), ((3,3),'Left') , ((3,2),'Left') ] )
#    print(wh.worker) # x,y  coords !!
#    print(wh.boxes)  # x,y  coords !!


if __name__ == "__main__":
    pass    
#    test_warehouse_1() # test Warehouse
#    test_warehouse_2() # test Warehouse
    
    print(my_team())  # should print your team

#    test_taboo_cells() 
#    test_check_elem_action_seq()
    test_solve_sokoban_elem()
#    test_can_go_there()
#    test_solve_sokoban_macro()   
