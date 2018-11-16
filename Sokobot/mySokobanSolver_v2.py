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




import search
import sokoban

from search import breadth_first_tree_search
from search import best_first_graph_search
from search import depth_limited_search
from search import astar_tree_search


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [ (n9840371, 'Sid hant', 'Mahapatra'), (n9160531,'Alec', 'Gurman')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):

    '''
    FIND TABOO CELLS

    Follow two simple rules:

        1. If a cell is a corner and not a target, then it is a taboo cell.

            # X .   .   X #
            # # # # # # # #

        2. All the cells between two corners along a wall are taboo if none of
             these cells is a target.

            # X X X X X X X #   # # #
            # # # # # # # # #   # X
                                # X
                                # X
                                # # #

    Checking the taboo cells follows the following method:

    1. if the coordinate is a wall, false
    2. if the coordinate is a corner, true
    3. if the coordinate is not a corner but is next to a wall true
    4. if is not corner but against wall but same x/y as target, False

    INPUTS:
        warhouse: warehouse object
    OUTPUTS:
        draw: string - visualize the taboo cells in the map

    '''

    X,Y = zip(*warehouse.walls)
    x_size, y_size = 1+max(X), 1+max(Y) #size of field

    vis = [[" "] * x_size for y in range(y_size)] #create an empty list

    for (x,y) in warehouse.walls: #draw the walls
        vis[y][x] = "#"

    for y in range(0 , len(vis)):
        for x in range(0 , len(vis[y])):
            if (x,y) not in warehouse.walls: #check if x,y is a wall
                if (x,y) not in warehouse.targets: #check if x,y is a target
                    #check if the coordinate is a corner
                    if ((x+1,y) in warehouse.walls and (x,y+1) in warehouse.walls) or ((x,y-1) in warehouse.walls and (x-1,y) in warehouse.walls) or ((x-1,y) in warehouse.walls and (x,y+1) in warehouse.walls) or ((x+1,y) in warehouse.walls and (x,y-1) in warehouse.walls):
                        vis[y][x] = "X"
                    #check columns for taboo  cells
                    elif ((x+1,y) in warehouse.walls) or ((x-1,y) in warehouse.walls):
                            if (x in [f[0] for f in warehouse.targets]):
                                vis[y][x] = " "
                            else:
                                vis[y][x] = "X"
                    #check rows for  taboo cells
                    elif ((x,y-1) in warehouse.walls) or ((x,y+1) in warehouse.walls):
                            if (y in [l[1] for l in warehouse.targets]):
                                vis[y][x] = " "
                            else:
                                vis[y][x] = "X"

    draw = "\n".join(["".join(line) for line in vis])
    return draw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SokobanPuzzle(search.Problem):
    '''
    Class to represent a Sokoban puzzle.
    Your implementation should be compatible with the
    search functions of the provided module 'search.py'.

    States are in the following format:
        ((worker_x, worker_y), (boxK_x,boxK_y))
        - Worker belongs in the 0 position
        -boxK inidcates an interating box value eg box1, box2, box2

    '''
    def __init__(self, warehouse):
        '''
        VARIABLES:
            inventory: warhouse object
            goal: warhouse targets
            initial: tuple, (worker,boxes)
            taboo: list of taboo cells

        INPUTS:
            warehouse: warehouse object
        '''
        self.inventory = warehouse
        self.goal = self.inventory.targets
        self.goal = tuple(self.goal)
        initial = list(self.inventory.boxes)
        initial.insert(0, self.inventory.worker) #Worked at position 0 in list
        self.initial = tuple(initial)
        self.taboo = self.getTabooCells()

    def goal_test(self, state):
        '''
        Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough.

        INPUTS:
            state: current global state
        OUTPUS:
            boolean, False if not a target otherwise True
        '''
        state = list(state)
        del state[0] #get rid of the worker as we are only checking the boxes
        for boxes in state:
            if boxes not in self.inventory.targets:
                return False
        print('\n------GOAL FOUND------\n')
        return True

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state
        if these actions do not push a box in a taboo cell.
        The actions must belong to the list ['Left', 'Down', 'Right', 'Up']

        INPUTS:
            state: current global state
        OUTPUTS:
            actions: list, current possible actions
        """

        state = list(state)
        available = list()

        (worker_x, worker_y) = state.pop(0) #removes the item at position 0 and returns it

        # Move the worker up/down/left/right if that spot is available and does not push a block into a taboo cell
        # if there are two blocks next to each other, cannot move that direction

        # For Upwards movement
        if ((worker_x,worker_y-1) not in self.inventory.walls):
            if ((worker_x,worker_y-1) in state):
                if ((worker_x,worker_y-2) not in self.inventory.walls) and \
                        ((worker_x,worker_y-2) not in state) and \
                        ((worker_x,worker_y-2) not in self.taboo):
                    available.append("Up")
            else:
                available.append("Up")
        # For Downwards Movement
        if ((worker_x,worker_y+1) not in self.inventory.walls):
            if ((worker_x,worker_y+1) in state):
                if ((worker_x,worker_y+2) not in self.inventory.walls) and \
                        ((worker_x,worker_y+2) not in state) and \
                        ((worker_x,worker_y+2) not in self.taboo):
                    available.append("Down")
            else:
                available.append("Down")
        #Towards Right
        if ((worker_x+1,worker_y) not in self.inventory.walls):
            if ((worker_x+1,worker_y) in state):
                if ((worker_x+2,worker_y) not in self.inventory.walls) and \
                        ((worker_x+2,worker_y) not in state) and \
                        ((worker_x+2,worker_y) not in self.taboo):
                    available.append("Right")
            else:
                available.append("Right")
        #Towards Left
        if ((worker_x-1,worker_y) not in self.inventory.walls):
            if ((worker_x-1,worker_y) in state):
                if ((worker_x-2,worker_y) not in self.inventory.walls) and \
                        ((worker_x-2,worker_y) not in state) and \
                        ((worker_x-2,worker_y) not in self.taboo):
                    available.append("Left")
            else:
                available.append("Left")


        return available

    def result(self, state, action):
        '''
        Update our worker  position along with any boxes it has pushed

        INPUTS:
            state: current global state
            action: list, avaialbe actions
        OUTPUTS:
            state: tuple, updated state
        '''

        # Move the worker in the direction, update any blocks it may push
        state = list(state)
        (worker_x, worker_y) = state.pop(0) #remove the 0th element and return it

        if action is 'Up':
            if (worker_x,worker_y-1) in state:
                    #move the box up one place
                    state[state.index((worker_x,worker_y-1))] = (worker_x,worker_y-2)
            #move the workere up one place
            state.insert(0, (worker_x,worker_y-1))
        elif action is 'Down':
            if (worker_x,worker_y+1) in state:
                    #move the box down one place
                    state[state.index((worker_x,worker_y+1))] = (worker_x,worker_y+2)
            #move the worker down one place
            state.insert(0, (worker_x,worker_y+1))
        elif action is 'Left':
            if (worker_x-1,worker_y) in state:
                    #move the box left one place
                    state[state.index((worker_x-1,worker_y))] = (worker_x-2,worker_y)
            #move the worker left one place
            state.insert(0, (worker_x-1,worker_y))
        elif action is 'Right':
            if (worker_x+1,worker_y) in state:
                    #move the box right one place
                    state[state.index((worker_x+1,worker_y))] = (worker_x+2,worker_y)
            #move the worker right one place
            state.insert(0, (worker_x+1,worker_y))
        else:
            raise ValueError("STATE NOT VALID")

        return tuple(state)

    def getTabooCells(self):

        '''
        GET TABOO CELLS

            - Same function as taboo_cells() but return a list of taboo cells

        OUTPUTS:
            taboo: list, taboo cell coordinates
        '''

        taboo = list()

        X,Y = zip(*self.inventory.walls)
        x_size, y_size = 1+max(X), 1+max(Y) #size of field
        vis = [[" "] * x_size for y in range(y_size)] #create an empty list
        for y in range(0 , len(vis)):
            for x in range(0 , len(vis[y])):
                if (x,y) not in self.inventory.walls: #check if x,y is a wall
                    if (x,y) not in self.inventory.targets: #check if x,y is a target
                        #check if the coordinate is a corner
                        if ((x+1,y) in self.inventory.walls and (x,y+1) in self.inventory.walls) or ((x,y-1) in self.inventory.walls and (x-1,y) in self.inventory.walls) or ((x-1,y) in self.inventory.walls and (x,y+1) in self.inventory.walls) or ((x+1,y) in self.inventory.walls and (x,y-1) in self.inventory.walls):
                            taboo.append((x,y))
                        #check columns for taboo  cells
                    elif ((x+1,y) in self.inventory.walls) or ((x-1,y) in self.inventory.walls):
                                if (x in [f[0] for f in self.inventory.targets]):
                                    pass
                                else:
                                    taboo.append((x,y))
                        #check rows for  taboo cells
                    elif ((x,y-1) in self.inventory.walls) or ((x,y+1) in self.inventory.walls):
                                if (y in [l[1] for l in self.inventory.targets]):
                                    pass
                                else:
                                    taboo.append((x,y))

        return taboo

    def path_cost(self, parent_cost, current_state, action, new_state):

        '''
        Calculates the path cost which for a elementary solver is just the parent path cost + 1
        
        OUTPUTS:
            path_cost: Numeric
        '''
        return parent_cost + 1

    def h(self, node):

        '''
        Calculate min distance between a box and closest goal
            - Returns a heuristic for total distance

        Based off a simple Manhattend distance heuristic model

        INPUTS:
            node: current node
        OUTPUTS:
            distance: distance between box and closest goals
        '''

        state = list(node.state)
        del state[0]
        distance = 0

        for i in xrange(len(state)):
            distance_box = []
            if state[i] not in self.inventory.targets:
                for j in xrange(len(self.inventory.targets)):
                    box_x, box_y = state[i]
                    target_x, target_y = self.inventory.targets[j]
                    distance_box_calc = abs(box_x - target_x) + abs(box_y - target_y)
                    distance_box.append(distance_box_calc)
                distance += min(distance_box)
        return distance

class SokobanPuzzleMacro(SokobanPuzzle):

    def check_adjacent(self, worker, box_state):
        '''
        Checks for adjacency in the following ways:

            1. Is worker adjecent to a box
            2. Is pushing the box possible
            3. Does pushing the box cause a deadlock

        A deadlock can be described in the following ways:

            1. # # # # #
               # $   $
               #   $     Pushing box down between two others will cause a lockup
               #   @
               #

            2. # # # # #
               # *       Pushing box to target is acceptable
               # . $ @
               #

        Our return value is a list of actions to push the box that satisfy the deadlock statements

        INPUTS:
            worker: worker coordinates
            box_state: state with only the box coordinates appended
        OUTPUTS:
            possible_actions: list, all possible actions with macro conditions
        '''

        worker_x, worker_y = worker
        possible_actions = []

        #For left movement
        if (worker_x-1,worker_y) in box_state:
            if (worker_x-2,worker_y) not in self.inventory.walls and \
                (worker_x-2,worker_y) not in box_state and \
                (worker_x-2,worker_y) not in self.taboo:

                if (worker_x-2,worker_y-1) not in box_state and \
                    (worker_x-2,worker_y+1) not in box_state:
                    possible_actions.append("Left")
                else:
                    deadlock = False
                    #box adjacent above
                    if (worker_x-2,worker_y-1) in box_state:
                        if (worker_x-3,worker_y-1) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x-2,worker_y-1) not in self.inventory.targets and \
                                (worker_x-2,worker_y) not in self.inventory.tartgets:
                                    deadlock = True
                    #box adjacent below
                    if (worker_x-2,worker_y+1) in box_state:
                        if (worker_x-3,worker_y+1) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x-2,worker_y+1) not in self.inventory.targets and \
                                (worker_x-2,worker_y) not in self.inventory.tartgets:
                                    deadlock = True
                    #if we aren't in a deadlock
                    if not deadlock:
                        possible_actions.append("Left")

        #For right movement
        if (worker_x+1,worker_y) in box_state:
            if (worker_x+2,worker_y) not in self.inventory.walls and \
                (worker_x+2,worker_y) not in box_state and \
                (worker_x+2,worker_y) not in self.taboo:

                if (worker_x+2,worker_y-1) not in box_state and \
                    (worker_x+2,worker_y+1) not in box_state:
                    possible_actions.append("Right")
                else:
                    deadlock = False
                    #box adjacent above
                    if (worker_x+2,worker_y-1) in box_state:
                        if (worker_x+3,worker_y-1) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x+2,worker_y-1) not in self.inventory.targets and \
                                (worker_x+2,worker_y) not in self.inventory.tartgets:
                                    deadlock = True
                    #box adjacent below
                    if (worker_x+2,worker_y+1) in box_state:
                        if (worker_x+3,worker_y+1) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x+2,worker_y+1) not in self.inventory.targets and \
                                (worker_x+2,worker_y) not in self.inventory.tartgets:
                                    deadlock = True
                    #if we aren't in a deadlock
                    if not deadlock:
                        possible_actions.append("Right")

        #For up movement
        if (worker_x,worker_y-1) in box_state:
            if (worker_x,worker_y-2) not in self.inventory.walls and \
                (worker_x,worker_y-2) not in box_state and \
                (worker_x,worker_y-2) not in self.taboo:

                if (worker_x-1,worker_y-2) not in box_state and \
                    (worker_x+1,worker_y-2) not in box_state:
                    possible_actions.append("Up")
                else:
                    deadlock = False
                    #box adjacent above
                    if (worker_x-1,worker_y-2) in box_state:
                        if (worker_x-1,worker_y-3) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x-1,worker_y-2) not in self.inventory.targets and \
                                (worker_x,worker_y-2) not in self.inventory.tartgets:
                                    deadlock = True
                    #box adjacent below
                    if (worker_x+1,worker_y-1) in box_state:
                        if (worker_x+1,worker_y-3) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x+1,worker_y-2) not in self.inventory.targets and \
                                (worker_x,worker_y-2) not in self.inventory.tartgets:
                                    deadlock = True
                    #if we aren't in a deadlock
                    if not deadlock:
                        possible_actions.append("Up")

        #For down movement
        print(worker_x,worker_y)

        if (worker_x,worker_y+1) in box_state:
            if (worker_x,worker_y+2) not in self.inventory.walls and \
                (worker_x,worker_y+2) not in box_state and \
                (worker_x,worker_y+2) not in self.taboo:

                if (worker_x-1,worker_y+2) not in box_state and \
                    (worker_x+1,worker_y+2) not in box_state:
                    possible_actions.append("Down")
                else:
                    deadlock = False
                    #box adjacent above
                    if (worker_x-1,worker_y+2) in box_state:
                        if (worker_x-1,worker_y+3) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x-1,worker_y+2) not in self.inventory.targets and \
                                (worker_x,worker_y+2) not in self.inventory.tartgets:
                                    deadlock = True
                    #box adjacent below
                    if (worker_x+1,worker_y+1) in box_state:
                        if (worker_x+1,worker_y+3) in self.inventory.walls and (worker_x-2,worker_y) in self.inventory.walls:
                            if (worker_x+1,worker_y+2) not in self.inventory.targets and \
                                (worker_x,worker_y+2) not in self.inventory.tartgets:
                                    deadlock = True
                    #if we aren't in a deadlock
                    if not deadlock:
                        possible_actions.append("Down")

        #print(possible_actions)
        return possible_actions

    def get_endpoints_aslist(self, boxes, worker, state):
        '''
        Simply finds the endpoint position in the warehouse which is the position
        recognised as adjacent to a box. Some macro endpoint may be impossible to complete
        so a check will be done to determine if they are possible.

        The return value is a list of the macro end points for a box that is moveable
        '''

        box_x, box_y = boxes
        worker_x, worker_y = worker

        macro_points = list()

        #check for walls and boxes right or left of box
        if (box_x-1,box_y) not in self.inventory.walls and (box_x-1,box_y) not in state and \
            (box_x+1,box_y) not in self.inventory.walls and (box_x+1,box_y) not in state:

            #Check proximity to box, if next to box, macro is not required
            if (box_x-1,box_y) is not (worker_x,worker_y):
                #Check for push right viabillity
                if (box_x+1,box_y) not in self.taboo:
                    macro_points.append((box_x-1,box_y)) #left
            if (box_x+1,box_y) is not (worker_x,worker_y):
                if (box_x-1,box_y) not in self.taboo:
                    macro_points.append((box_x+1,box_y)) #right

        #check for walls and boxes up or down of box
        if ((box_x,box_y-1) not in self.inventory.walls) and ((box_x,box_y-1) not in state) and \
            ((box_x,box_y+1) not in self.inventory.walls) and ((box_x,box_y+1) not in state):
            #Check proximity to box, if next to box, macro is not required
            if (box_x,box_y-1) is not (worker_x,worker_y):
                #Check for push right viabillity
                if(box_x,box_y+1) not in self.taboo:
                    macro_points.append((box_x,box_y-1)) #left
            if (box_x,box_y+1) is not (worker_x,worker_y):
                if (box_x,box_y-1) not in self.taboo:
                    macro_points.append((box_x,box_y+1)) #right

        return macro_points

    def macro_actions(self, macro_points, worker, state):
        '''
        Gets the path from worker to macro endpoint using our a* search and distance heuristic.

            - Return value is the path of the a*star search procedure
        '''

        aux_problem = MacroHelperShortestPath(worker, self.inventory.walls, state, macro_points)
        node = astar_tree_search(aux_problem,aux_problem.h)
        return node

    def get_macro_actions_full(self, node):
        actions = []
        if node is None:
            return None
        else:
            for node in node.path():
                if node.action is not None:
                    actions.append(node.action)
            return actions

    def actions(self, state):
        """
        Get possible actions
            1. Checks if a worker is adjacent to a box and determines if pushing
            that box is possible
            2. Identifies possible macro end points for every box and obtains
            the optimal actions in order to move the worker to the end point
        """
        state = list(state)
        actions = list()

        worker_x, worker_y = state.pop(0) #remove the 0th element and return it

        # Check for worker adjacency to a box and pass any possible actions
        actions_boxes = self.check_adjacent((worker_x, worker_y), state)
        # add possible actions
        if actions_boxes:
            for action in actions_boxes:
                actions.append(action)
        for boxes in state:
            macro_endpoints = self.get_endpoints_aslist(boxes, (worker_x, worker_y), state)
            # solve the auxilary problem for the macro end points
            if macro_endpoints:
                for target in macro_endpoints:
                    macro_action = self.macro_actions(target, (worker_x, worker_y), state)
                    if macro_action is not None:
                        actions.append(self.get_macro_actions_full(macro_action))
        return actions

    def result(self, state, action):
        """
        Move the worker in the new direction and update any boxes it pushes
        """

        state = list(state)
        worker_x, worker_y = state.pop(0) #get the 0th element then remove it

        # check if the given action is a list (i.e it a macro action)
        if isinstance(action, list):
            for sub_move in action:
                if sub_move == 'Left':
                    worker_x -= 1
                elif sub_move == 'Right':
                    worker_x += 1
                elif sub_move == 'Up':
                    worker_y -= 1
                elif sub_move == 'Down':
                    worker_y += 1
                else:
                    raise ValueError("SUB MOVE IS NOT VALID")
            state.insert(0, (worker_x, worker_y))
        else:
            # our action is not a macro so just do a single action move
            if action is 'Up':
                if (worker_x,worker_y-1) in state:
                        #move the box up one place
                        state[state.index((worker_x,worker_y-1))] = (worker_x,worker_y-2)
                #move the workere up one place
                state.insert(0, (worker_x,worker_y-1))
            elif action is 'Down':
                if (worker_x,worker_y+1) in state:
                        #move the box down one place
                        state[state.index((worker_x,worker_y+1))] = (worker_x,worker_y+2)
                #move the worker down one place
                state.insert(0, (worker_x,worker_y+1))
            elif action is 'Left':
                if (worker_x-1,worker_y) in state:
                        #move the box left one place
                        state[state.index((worker_x-1,worker_y))] = (worker_x-2,worker_y)
                #move the worker left one place
                state.insert(0, (worker_x-1,worker_y))
            elif action is 'Right':
                if (worker_x+1,worker_y) in state:
                        #move the box right one place
                        state[state.index((worker_x+1,worker_y))] = (worker_x+2,worker_y)
                #move the worker right one place
                state.insert(0, (worker_x+1,worker_y))
            else:
                raise ValueError("STATE NOT VALID")

        return tuple(state)

    def path_cost(self, parent_cost, current_state, action, new_state):
        '''
        Get path cost
            Macro: increased with amount of actions
            Single: parent_cost + 1
        '''
        if isinstance(action,list):
            return len(action) + parent_cost
        return parent_cost + 1


class MacroHelperShortestPath(search.Problem):
    """

    Auxilary Problem class of the Sokoban puzzle
    This class allows us the find the shortest path from a worker to a goal through
    macro action endpoints. It is a helper class for our SokobanMacroSolver

    """

    def __init__(self, worker, walls, boxes, goal):

        self.initial = worker
        self.walls = walls
        self.boxes = boxes
        self.goal = goal

    def actions(self, state):

        actions = list()
        (worker_x, worker_y) = state

        #check if worker can move left
        if (worker_x-1, worker_y) not in self.walls and (worker_x-1, worker_y) not in self.boxes:
                actions.append('Left')
        #check if worker can move right
        if (worker_x+1, worker_y) not in self.walls and (worker_x+1, worker_y) not in self.boxes:
                actions.append('Right')
        #check if worker can move down
        if (worker_x, worker_y+1) not in self.walls and (worker_x, worker_y+1) not in self.boxes:
                actions.append('Down')
        #check if worker can move up
        if (worker_x, worker_y-1) not in self.walls and (worker_x, worker_y-1) not in self.boxes:
                actions.append('Up')

        return actions

    def result(self, state, action):

        (worker_x, worker_y) = state
        if action is 'Left':
            return worker_x-1, worker_y
        elif action is 'Right':
            return worker_x+1, worker_y
        elif action is 'Up':
            return worker_x, worker_y-1
        elif action is 'Down':
            return worker_x, worker_y+1
        else:
            raise ValueError("ACTION IS INVALID")

    def goal_test(self, state):

        return state == self.goal


    def h(self, node):

        (worker_x, worker_y) = node.state
        (target_x, target_y) = self.goal
        distance = abs(worker_x - target_x) + abs(worker_y - target_y)
        return distance

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
    '''

    duplicate = warehouse.copy(warehouse.worker , warehouse.boxes) #only positions of worker and boxes are changing.
    duplicate.worker = list(duplicate.worker) # force conversion from tuple to list
    for r,item in enumerate(duplicate.boxes):
        duplicate.boxes[r] = list(duplicate.boxes[r])
    for move in action_seq:
        if(move is 'Down'):
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
        elif(move is 'Up'):
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

    problem = SokobanPuzzle(warehouse)
    steps = astar_tree_search(problem, problem.h)
    actions = list()

    if steps is None:
        return ['Impossible']
    else:
        for node in steps.path():
            if node.action is not None:
                #check if the path of actions is a list (more than one element)
                if isinstance(node.action, list):
                    for action in node.action:
                        actions.append(action)
                else:
                    actions.append(node.action)
    return actions

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

    problem = SokobanPuzzleMacro(warehouse)
    steps = astar_tree_search(problem, problem.h)
    actions = list()

    if steps is None:
        return ['Impossible']
    else:
        for node in steps.path():
            if node.action is not None:
                #check if the path of actions is a list (more than one element)
                if isinstance(node.action, list):
                    for action in node.action:
                        actions.append(action)
                else:
                    actions.append(node.action)
    return actions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
