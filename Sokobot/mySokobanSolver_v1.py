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



'''
Last Modified: Sid Mahapatra [4-24-2017] 
'''

import search

import sokoban

from search import breadth_first_tree_search
from search import breadth_first_graph_search
from search import depth_limited_search
from search import astar_tree_search
from sokoban import Warehouse

#import random

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ ('n9840371', 'Sid', 'Mahapatra'), ('n9160531','Alec', 'Gurman')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell is called 'taboo' 
    if whenever a box get pushed on such a cell then the puzzle becomes unsolvable.  
    When determining the taboo cells, you must ignore all the existing boxes, 
    simply consider the walls and the target  cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: a Warehouse object
    @return
       A string representing the puzzle with only the wall cells marked with 
       an '#' and the taboo cells marked with an 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    X,Y = zip(*warehouse.walls)
    x_size, y_size = 1+max(X), 1+max(Y)
        
    vis = [[" "] * x_size for y in range(y_size)]
    for (x,y) in warehouse.walls:
        vis[y][x] = "#"
    #print(vis)
    for y in range(0 , len(vis)):
    	#print(y)
        for x in range(0 , len(vis[y])):
            #print(x)
            if is_taboo(x,y,warehouse):
                vis[y][x] = 'X'
                
    if 'X' in vis[0]:
        vis[0].clear()
    #print(vis)
    #print(warehouse.walls) #list of tuples (delete comment)
    return "\n".join(["".join(line) for line in vis]) # from sokoban.py

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    Class to represent a Sokoban puzzle.
    Your implementation should be compatible with the
    search functions of the provided module 'search.py'.
    
    	Use the sliding puzzle and the pancake puzzle for inspiration!
    
    '''
    def __init__(self, warehouse, initial = None, goal = None):
        self.inventory = warehouse
        self.inventory.worker = list(self.inventory.worker) # force conversion from tuple to list
        for r,item in enumerate(self.inventory.boxes):
    	    self.inventory.boxes[r] = list(item)    	
        X,Y = zip(*warehouse.walls)
        self.x_max, self.y_max = 1+max(X), 1+max(Y)
        if goal is None:
            self.goal = self.inventory.targets
        else:
            #assert goal == self.inventory.targets
            self.goal = goal
        if initial is None:
            self.initial = self.inventory.boxes
        else:
            self.initial = initial # current positions of the boxes
        self.initial = tuple(self.initial)
        self.goal = tuple(self.goal)

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state 
        if these actions do not push a box in a taboo cell.
        The actions must belong to the list ['Left', 'Down', 'Right', 'Up']        
        """
        self.inventory.boxes = state
        available = []
        # Move the worker up/down/left/right if that spot is available and does not push a block into a taboo cell
        # if there are two blocks next to each other, cannot move that direction
        
        # For Upwards movement
        if ((self.inventory.worker[0],self.inventory.worker[1]+1) in self.inventory.walls):
        	pass
        elif ((self.inventory.worker[0],self.inventory.worker[1]+1) in self.inventory.boxes):
            if (((self.inventory.worker[0],self.inventory.worker[1]+2) in self.inventory.walls) or ((self.inventory.worker[0],self.inventory.worker[1]+2) in self.inventory.boxes)):
                pass
            elif (not is_taboo(self.inventory.worker[0],self.inventory.worker[1]+2,self.inventory)):
                available.append("Up")
        else:
            available.append("Up")
        # For Downwards Movement
        if ((self.inventory.worker[0],self.inventory.worker[1]-1) in self.inventory.walls):
        	pass
        elif ((self.inventory.worker[0],self.inventory.worker[1]-1) in self.inventory.boxes):
            if (((self.inventory.worker[0],self.inventory.worker[1]-2) in self.inventory.walls) or ((self.inventory.worker[0],self.inventory.worker[1]-2) in self.inventory.boxes)):
                pass
            elif (not is_taboo(self.inventory.worker[0],self.inventory.worker[1]-2,self.inventory)):
                available.append("Down")
        else:
            available.append("Down")
        #Towards Right
        if ((self.inventory.worker[0]+1,self.inventory.worker[1]) in self.inventory.walls):
        	pass
        elif ((self.inventory.worker[0]+1,self.inventory.worker[1]) in self.inventory.boxes):
            if (((self.inventory.worker[0]+2,self.inventory.worker[1]) in self.inventory.walls) or ((self.inventory.worker[0]+2,self.inventory.worker[1]) in self.inventory.boxes)):
                pass
            elif (not is_taboo(self.inventory.worker[0]+2,self.inventory.worker[1],self.inventory)):
                available.append("Right")
        else:
            available.append("Right")
        #Towards Left
        if ((self.inventory.worker[0]-1,self.inventory.worker[1]) in self.inventory.walls):
        	pass
        elif ((self.inventory.worker[0]-1,self.inventory.worker[1]) in self.inventory.boxes):
            if (((self.inventory.worker[0]-2,self.inventory.worker[1]) in self.inventory.walls) or ((self.inventory.worker[0]-2,self.inventory.worker[1]) in self.inventory.boxes)):
                pass
            elif (not is_taboo(self.inventory.worker[0]-2,self.inventory.worker[1],self.inventory)):
                available.append("Left")
        else:
            available.append("Left")
            
  
        return available
        
    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""
        print(self.inventory.targets)
        print(state)
        print(self.inventory.boxes)
        if state is self.inventory.targets:
            print ('success')
            return True

    def result(self, state, action):
        
        # Move the worker in the direction, update any blocks it may push
        self.inventory.boxes = list(state)
        self.inventory.worker = list(self.inventory.worker) # force conversion from tuple to list
        for r,item in enumerate(self.inventory.boxes):
    	    self.inventory.boxes[r] = list(self.inventory.boxes[r])
        if action is 'Up':
        	for i,box in enumerate(self.inventory.boxes):
        		if box is (self.inventory.worker[0],self.inventory.worker[1]+1):
        			self.inventory.boxes[i][1] = box[1]+1
        	self.inventory.worker[1] = self.inventory.worker[1] + 1
        elif action is 'Down':
        	for i,box in enumerate(self.inventory.boxes):
        		if box is (self.inventory.worker[0],self.inventory.worker[1]-1):
        			self.inventory.boxes[i][1] = box[1]-1
        	self.inventory.worker[1] = self.inventory.worker[1] - 1
        elif action is 'Left':
        	for i,box in enumerate(self.inventory.boxes):
        		if box is (self.inventory.worker[0]-1,self.inventory.worker[1]):
        			self.inventory.boxes[i][0] = box[0]-1
        	self.inventory.worker[0] = self.inventory.worker[0] - 1
        elif action is 'Right':
        	for i,box in enumerate(self.inventory.boxes):
        		if box is (self.inventory.worker[0]+1,self.inventory.worker[1]):
        			self.inventory.boxes[i][0] = box[0]+1
        	self.inventory.worker[0] = self.inventory.worker[0] + 1
        state = tuple(self.inventory.boxes)
        return state
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def is_taboo(x, y, warehouse):
    # if the coordinate is a wall, false
    # if the coordinate is a corner, true
    # if the coordinate is not a corner but is next to a wall true
    # if is not corner but against wall but same x/y as target, False
    
    if (x,y) in warehouse.walls:
            return False
    elif (x,y) in warehouse.targets:
            return False
    elif ((x+1,y) in warehouse.walls and (x,y+1) in warehouse.walls) or ((x,y-1) in warehouse.walls and (x-1,y) in warehouse.walls)or ((x-1,y) in warehouse.walls and (x,y+1) in warehouse.walls) or ((x+1,y) in warehouse.walls and (x,y-1) in warehouse.walls):
            return True
    elif ((x+1,y) in warehouse.walls) or ((x-1,y) in warehouse.walls): 
            if (x in [f[0] for f in warehouse.targets]):
                return False
    elif ((x,y-1) in warehouse.walls) or ((x,y+1) in warehouse.walls):
            if (y in [l[1] for l in warehouse.targets]):
            	return False
            return True
    return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object
    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
               Note: change 'Down to Up' in tester script
    '''
    duplicate = warehouse.copy(warehouse.worker , warehouse.boxes) #only positions of worker and boxes are changing.
    duplicate.worker = list(duplicate.worker) # force conversion from tuple to list
    for r,item in enumerate(duplicate.boxes):
    	duplicate.boxes[r] = list(duplicate.boxes[r])
    for move in action_seq:
        if(move is 'Up'):
            if (duplicate.worker[0],duplicate.worker[1]+1) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0],duplicate.worker[1]+1) in duplicate.boxes and (duplicate.worker[0],duplicate.worker[1]+2) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0],duplicate.worker[1]+1) in duplicate.boxes and (duplicate.worker[0],duplicate.worker[1]+2) in duplicate.boxes:
                return "Failure"
            else:
                for i,box in enumerate(duplicate.boxes):
                    if box is (duplicate.worker[0],duplicate.worker[1]+1):
                        duplicate.boxes[i][1] = box[1]+1
                duplicate.worker[1] = duplicate.worker[1]+1
        elif(move is 'Down'):
            if (duplicate.worker[0],duplicate.worker[1]-1) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0],duplicate.worker[1]-1) in duplicate.boxes and (duplicate.worker[0],duplicate.worker[1]-2) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0],duplicate.worker[1]-1) in duplicate.boxes and (duplicate.worker[0],duplicate.worker[1]-2) in duplicate.boxes:
                return "Failure"
            else:
                for i,box in enumerate(duplicate.boxes):
                    if box is (duplicate.worker[0],duplicate.worker[1]-1):
                        duplicate.boxes[i][1] = box[1]-1
                duplicate.worker[1]= duplicate.worker[1]-1
        elif(move is 'Right'):
            if (duplicate.worker[0]+1,duplicate.worker[1]) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0]+1,duplicate.worker[1]) in duplicate.boxes and (duplicate.worker[0]+2,duplicate.worker[1]) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0]+1,duplicate.worker[1]) in duplicate.boxes and (duplicate.worker[0]+2,duplicate.worker[1]) in duplicate.boxes:
                return "Failure"
            else:
                for i,box in enumerate(duplicate.boxes):
                    if box is (duplicate.worker[0]+1,duplicate.worker[1]):
                        duplicate.boxes[i][0] = box[0]+1
                duplicate.worker[0]= duplicate.worker[0]+1
        elif(move is 'Left'):
            if (duplicate.worker[0]-1,duplicate.worker[1]) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0]-1,duplicate.worker[1]) in duplicate.boxes and (duplicate.worker[0]-2,duplicate.worker[1]) in warehouse.walls:
                return "Failure"
            elif (duplicate.worker[0]-1,duplicate.worker[1]) in duplicate.boxes and (duplicate.worker[0]-2,duplicate.worker[1]) in duplicate.boxes:
                return "Failure"
            else:
                for i,box in enumerate(duplicate.boxes):
                    if box is (duplicate.worker[0]-1,duplicate.worker[1]):
                        duplicate.boxes[i][0] = box[0]-1
                duplicate.worker[0]= duplicate.worker[0]-1
                
    
    return duplicate.__str__()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using elementary actions 
    the puzzle defined in a file.
    
    @param warehouse: a valid Warehouse object
    @return
        A list of strings.
        If puzzle cannot be solved return ['Impossible']
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []

    '''
    #if SokobanPuzzle(warehouse).goal_test(warehouse.boxes):
    #   return []
    
    steps = breadth_first_tree_search(SokobanPuzzle(warehouse))
    
    #if check_action_seq(warehouse, steps.actions()) is "Failure":
    #    return "Impossible"
    return steps
    '''
    state = warehouse.boxes
    puzzle = SokobanPuzzle(warehouse)
    
    if puzzle.goal_test(state):
        return []
    
    solution = breadth_first_tree_search(puzzle)
    
    #if check_action_seq(warehouse, solution.actions()) is "Failure":
    #    return "Impossible"
    return solution    
    '''

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,col) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object
    @return
      True if the worker can walk to cell dst=(row,col) without pushing any box
      False otherwise
    '''
    X,Y = zip(*warehouse.walls)
    x_size, y_size = 1+max(X), 1+max(Y)
    
    for obs in warehouse.boxes:
        if (dst[0]>x_size or dst[0]<0 or dst[1]>y_size or dst[1]<0):
            return False
        elif obs[1] in (range(warehouse.worker[1],dst[1]) or range(dst[1],warehouse.worker[1])) and obs[0] is warehouse.worker[0]:
            return False
        elif obs[0] in (range(warehouse.worker[0],dst[0]) or range(dst[0],warehouse.worker[0])) and obs[1] is warehouse.worker[1]:
            return False
        #elif obs[0] in range(warehouse.worker[0],dst[0]) and obs[1] in range(warehouse.worker[1],dst[1]):
         #   return False # Not sure about this! (delete comment)
        return True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    '''    
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    @param warehouse: a valid Warehouse object
    @return
        If puzzle cannot be solved return ['Impossible']
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
    
    # use the actions defined for elementary actions but change all actions that do not push a block
    # into the coordinate that would result from those actions

    macro_actions = []
    
    duplicate = warehouse.copy(warehouse.worker , warehouse.boxes)
    elementary =  solve_sokoban_elem(warehouse) #['Right','Right','Right','Down','Left','Left']
    duplicate.worker = list(duplicate.worker)
    for r,item in enumerate(duplicate.boxes):
    	duplicate.boxes[r] = list(duplicate.boxes[r])        
    #print(duplicate.boxes)
    #print(duplicate.worker)
    if elementary is "Impossible":
        return elementary
    
    for action in elementary:
        if action is 'Up':
            for i,box in enumerate(duplicate.boxes):
                if duplicate.boxes[i][1] is duplicate.worker[1]+1 and duplicate.boxes[i][0] is duplicate.worker[0]:
                   duplicate.boxes[i][1] = box[1]+1
                   macro_actions.append((duplicate.worker,'Up'))
            duplicate.worker[1] = duplicate.worker[1] + 1
        if action is 'Down':
            for i,box in enumerate(duplicate.boxes):
                if duplicate.boxes[i][1] is duplicate.worker[1]-1 and duplicate.boxes[i][0] is duplicate.worker[0]:
                   duplicate.boxes[i][1] = box[1]-1
                   macro_actions.append((duplicate.worker,'Down'))
            duplicate.worker[1] = duplicate.worker[1] - 1
        if action is 'Right':
            for i,box in enumerate(duplicate.boxes):
                if duplicate.boxes[i][0] is duplicate.worker[0]+1 and duplicate.boxes[i][1] is duplicate.worker[1]:
                   duplicate.boxes[i][0] = box[0]+1
                   macro_actions.append((duplicate.worker,'Right'))
            duplicate.worker[0] = duplicate.worker[0] + 1
        if action is 'Left':
            for i,box in enumerate(duplicate.boxes):
                if duplicate.boxes[i][0] is duplicate.worker[0]-1 and duplicate.boxes[i][1] is duplicate.worker[1]:
                   duplicate.boxes[i][0] = box[0]-1
                   macro_actions.append((duplicate.worker,'Left'))
            duplicate.worker[0] = duplicate.worker[0] - 1

    
    return macro_actions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def test_check_macro_action_seq(warehouse, actions):
    raise NotImplementedError()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -