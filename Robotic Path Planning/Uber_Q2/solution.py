"""Copyright [2019] [Siddhant Mahapatra]

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



import sys
import fractions

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxint
        # Mark all nodes unvisited        
        self.visited = False  
        # Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

import heapq

def dijkstra(aGraph, start):
    print '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero 
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


if __name__ == '__main__':

    g = Graph()

    arr = []
    n_cities = int(input("Enter the number of cities: "))
    z = 0
    while z < n_cities:
        r = int(input("Enter Number %s: " %(z+1)))
        arr.append(r)
        z = z + 1

    t = int(input("Enter Threshold: "))

    for q in arr:
        g.add_vertex(q)

    #g.add_vertex(1)
    #g.add_vertex(2)
    #g.add_vertex(3)
    #g.add_vertex(4)
    #g.add_vertex(5)
    #g.add_vertex(6)

    l = 0
    test = []
    for m in arr:
        for o in arr[:]:
            if m == o:
                continue
            l = fractions.gcd(m, o)
            if l > t:
                g.add_edge(m,o,1)
    
    #g.add_edge(1, 3, 1)
    #g.add_edge(1, 6, 1)
    #g.add_edge(2, 3, 1)
    #g.add_edge(2, 4, 1)
    #g.add_edge(3, 4, 1)
    #g.add_edge(3, 6, 1)
    #g.add_edge(4, 5, 1)
    #g.add_edge(5, 6, 1)

    print 'Graph data:'
    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print '( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w))

    x = int(input("Enter the Number of Origin-Destination pairs: "))
    z = 0
    p = []
    while z < x:
        k = int(input("Enter Origin %s : "%(z+1)))
        h = int(input("Enter Destination %s : "%(z+1)))
        z = z + 1
        dijkstra(g, g.get_vertex(k)) 
        target = g.get_vertex(h)
        path = [target.get_id()]
        shortest(target, path)
        p.append(path[::-1])
        g = Graph()
        for q in arr:
            g.add_vertex(q)
        l = 0
        test = []
        for m in arr:
            for o in arr[:]:
                if m == o:
                    continue
                l = fractions.gcd(m, o)
                if l > t:
                    g.add_edge(m,o,1)

    print "Shortest Paths: %s" %p
    t_f_result = []
    for a in p:
        if len(a)<2:
            t_f_result.append(0)
        else:
            t_f_result.append(1)
    print "Final Result in terms of Boolean Array: %s" %(t_f_result)

    """
    dijkstra(g, g.get_vertex(3)) 

    target = g.get_vertex(6)
    path = [target.get_id()]
    shortest(target, path)
    #print 'The shortest path : %s' %(path[::-1])
    """