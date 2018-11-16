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


import numpy as np
import matplotlib.pyplot as plt
import math
import pprint

def dijkstras(occupancy_map, x_spacing, y_spacing, start, goal):
	DEBUG = False
    VISUAL = True
    colormapval = (0, 8)
    goal_found = False

    # Setup Map Visualizations:
    if VISUAL == True:
        viz_map=occupancy_map
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        ax.set_title('Occupancy Grid')
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        plt.imshow(viz_map, origin='upper', interpolation='none', clim=colormapval)
        ax.set_aspect('equal')
        plt.pause(2)
    # We will use this delta function to search surrounding nodes.
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right
    cost = 1
    # Converting numpy array of map to list of map.
    occ_map = occupancy_map.tolist()
    if DEBUG == True:
        print "occ_map: "
        pprint.pprint(occ_map)

    # Converge start and goal positions to map indices.
    x = int(math.ceil((start.item(0) / x_spacing) - 0.5))  # startingx
    y = int(math.ceil((start.item(1) / y_spacing) - 0.5))  # startingy
    goalX = int(math.ceil((goal.item(0) / x_spacing) - 0.5))
    goalY = int(math.ceil((goal.item(1) / y_spacing) - 0.5))
    print "Start Pose: ", x, y
    print "Goal Pose: ", goalX, goalY

    # Make a map to keep track of all the nodes and their true cost values.
    possible_nodes = [[0 for row in range(len(occ_map[0]))] for col in range(len(occ_map))] 
    row = y
    col = x

    # Show the starting node and goal node.
    # 5 looks similar to S and 6 looks similar to G.
    possible_nodes[row][col] = 5

    if VISUAL == True:
        viz_map[row][col] = 5
        viz_map[goalY][goalX] = 6
        plt.imshow(viz_map, origin='upper', interpolation='none', clim=colormapval)
        plt.pause(2)

    if DEBUG == True:
        print "Possible Nodes: "
        pprint.pprint(possible_nodes)

    # g_value will count the number of steps each node is from the start.
    # Since I am at the start node, the total cost is zero.
    g_value = 0
    frontier_nodes = [(g_value, col, row)] # dist, x, y
    searched_nodes = []
    parent_node = {}  # NOTE : Dictionary that Maps {child node : parent node}
    loopcount = 0

    while len(frontier_nodes) != 0:
        if DEBUG == True:
            "\n>>>>>>>>>>>>LOOP COUNT: ", loopcount, "\n"
        frontier_nodes.sort(reverse=True) #sort from shortest distance to farthest
        current_node = frontier_nodes.pop()
        if DEBUG == True:
            print "current_node: ", current_node
            print "frontier nodes: ", searched_nodes
        if current_node[1] == goalX and current_node[2] == goalY:
            print "Goal found!"
            goal_found = True
            if VISUAL == True:
                plt.text(2, 10, s="Goal found!", fontsize=18, style='oblique', ha='center', va='top')
                plt.imshow(viz_map, origin='upper', interpolation='none', clim=colormapval)
                plt.pause(2)
            break
        g_value, col, row = current_node

        # Check surrounding neighbors.
        for i in delta:
            possible_expansion_x = col + i[0]
            possible_expansion_y = row + i[1]
            valid_expansion = 0 <= possible_expansion_y < len(occupancy_map[0]) and 0 <= possible_expansion_x < len(occ_map)
            if DEBUG == True:
                print "Current expansion Node: ", possible_expansion_x, possible_expansion_y











