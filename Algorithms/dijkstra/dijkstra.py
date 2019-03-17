#!/usr/bin/env python

import collections, string, random, sys
from collections import *

defaultDist = float("inf") # assuming distances initially undefined (inf)

class Edge(object):
    def __init__(self,sourceNode,destinationNode,cost):
        self.sourceNode = sourceNode
        self.destinationNode = destinationNode
        self.cost = cost

    # method to create itself (Edge)
    def create(self):
        return Edge(self.sourceNode,self.destinationNode,self.cost)

class Network(object):
    def __init__(self,edges):
        # isValid confirmation
        # confirming if all edges are in valid form
        try:
            self.edges = [edge.create() for edge in edges]
        except Exception as e:
            raise ValueError
    # generates vertices of network - creates self.vertices
    @property
    def vertices(self):
        verts = set()
        for edge in self.edges:
            verts.add(edge.sourceNode)
            verts.add(edge.destinationNode)
        return verts
    # nearest neighbors of each node
    @property
    def allClosestNodes(self):
        closestNodes = dict()
        for vertex in self.vertices:
            closestNodes[vertex] = set()
        for edge in self.edges:
            otherEnd = (edge.destinationNode, edge.cost)
            closestNodes[edge.sourceNode].add(otherEnd)
        return closestNodes
    # finding pair connections in network
    def getNodePairs(n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs
    # initializes distances to be infinite & previous vertices - None
    def distsVerts(self):
        distances = dict()
        for vertex in self.vertices:
            distances[vertex] = defaultDist
        priorVertices = dict()
        for vertex in self.vertices:
            priorVertices[vertex] = None
        return distances,priorVertices
    # current vertex should be node closest to the visited nodes
    def getCurrVertex(self,distances,visitNodes):
        minDist = float("inf"); currVertex = list(visitNodes)[0]
        for vertex in visitNodes:
            if distances[vertex] < minDist:
                minDist = distances[vertex]
                currVertex = vertex
        return currVertex
    # implements dijkstra's, checking validity 
    def dijkstra(self, origin, destination):
        assert(origin in self.vertices),"Source Node Not A Vertex"
        distances, priorVertices = self.distsVerts()
        distances[origin] = 0
        visitNodes = self.vertices.copy()
        # while not all nodes have been visited, take closest nodes to visited
        # nodes & find cheapest path from currVertex
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
        # construct path from prior vertices
        path, currVertex = [], destination
        while priorVertices[currVertex] != None:
            path = [currVertex] + path
            currVertex = priorVertices[currVertex]
        if len(path) > 0:
            path = [currVertex] + path
        return path
