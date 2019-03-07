#!/usr/bin/env python

import collections, string, random, sys
from collections import *

defaultDist = float("inf") # assuming distances initially undefined (inf)

class Edge(object):
    def __init__(self,sourceNode,destinationNode,cost):
        self.sourceNode = sourceNode
        self.destinationNode = destinationNode
        self.cost = cost

    def create(self):
        return Edge(self.sourceNode,self.destinationNode,self.cost)

class Network(object):
    def __init__(self,edges):
        # isValid confirmation
        # confirming if all edges are in valid form
        try:
            self.edges = [edge.create() for edge in edges]
        except:
            raise ValueError
    @property
    def vertices(self):
        verts = set()
        for edge in self.edges:
            verts.add(edge.sourceNode)
            verts.add(edge.destinationNode)
        return verts
    @property
    def allClosestNodes(self):
        closestNodes = dict()
        for vertex in self.vertices:
            closestNodes[vertex] = set()
        for edge in self.edges:
            otherEnd = (edge.destinationNode, edge.cost)
            closestNodes[edge.sourceNode].add(otherEnd)
        return closestNodes

    def getNodePairs(n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs
    def distsVerts(self):
        distances = dict()
        for vertex in self.vertices:
            distances[vertex] = defaultDist
        priorVertices = dict()
        for vertex in self.vertices:
            priorVertices[vertex] = None
        return distances,priorVertices
    def getCurrVertex(self,distances,visitNodes):
        minDist = float("inf")
        for vertex in visitNodes:
            if distances[vertex] < minDist:
                minDist = distances[vertex]
                currVertex = vertex
        return currVertex
    def dijkstra(self, origin, destination):
        assert(origin in self.vertices),"Source Node Not A Vertex"
        distances, priorVertices = self.distsVerts()
        distances[origin] = 0
        visitNodes = self.vertices.copy()

        while len(visitNodes)>0:
            currVertex = self.getCurrVertex(distances,visitNodes)
            if distances[currVertex] == defaultDist:
                break
            for closeNode, cost in self.allClosestNodes[currVertex]:
                otherPath = distances[currVertex] + cost
                if otherPath < distances[closeNode]:
                    distances[closeNode] = otherPath
                    priorVertices[closeNode] = currVertex
            visitNodes.remove(currVertex)

        path, currVertex = [], destination
        while priorVertices[currVertex] != None:
            path = [currVertex] + path
            currVertex = priorVertices[currVertex]
        if len(path) > 0:
            path = [currVertex] + path
        return path

if __name__ == "__main__":
    letters = string.ascii_uppercase
    L = []
    items = []
    outerLoop,innerLoop = int(sys.argv[1]),int(sys.argv[2])
    for source in range(outerLoop):
        for dest in range(source+1,innerLoop):
            r = random.randint(1,20)
            items += [letters[source],letters[dest]]
            L += [Edge(letters[source],letters[dest],r)]
            r = random.randint(1,20)
            items += [letters[dest],letters[source]]
            L += [Edge(letters[dest],letters[source],r)]
    graph = Network(L)

    start = random.randint(0,len(L)-1)
    end = random.randint(0,len(L)-1)
    print(graph.dijkstra(L[start].sourceNode,L[end].sourceNode))
