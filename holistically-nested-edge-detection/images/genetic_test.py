from matplotlib import pyplot as plt
import random
import getopt
import sys

class Graph:
	# takes a list of nodes, converts it into a dict
	def __init__(self, nodes):
		self.nodes = {}
		self.connections = {}

		for node in nodes:
			self.addNode(node)

	def addNode(self, node):
		if not (node in self.nodes.keys()):
			self.nodes[node] = []

	def addConnection(self, n1, n2, weight):
		if (n1 in self.nodes.keys()) and (n2 in self.nodes.keys()):
			self.connections[(n1, n2)] = weight

	def find(self, parent, i):
		if parent[i] == i:
			return i
		return self.find(parent, parent[i])

	def apply_union(self, parent, rank, x, y):
		xroot = self.find(parent, x)
		yroot = self.find(parent, y)
		if rank[xroot] < rank[yroot]:
			parent[xroot] = yroot
		elif rank[xroot] > rank[yroot]:
			parent[yroot] = xroot
		else:
			parent[yroot] = xroot
			rank[xroot] += 1

	def kruskals(self):
		result = []
		currentIndex = 0
		parent = []
		rank = []

		sortedEdges = sorted(self.connections.keys(), key=lambda x: self.connections[x])

		allNodes = list(self.nodes.keys())

		for node in range(len(self.nodes)):
			parent.append(node)
			rank.append(0)

		while len(result) < len(self.nodes.keys()) - 1:
			currentIndex += 1

			currentEdge = sortedEdges[currentIndex]
			v1 = allNodes.index(currentEdge[0])
			v2 = allNodes.index(currentEdge[1])

			v1Set = self.find(parent, v1)
			v2Set = self.find(parent, v2)

			if v1Set != v2Set:
				result.append(currentEdge)
				self.apply_union(parent, rank, v1Set, v2Set)
		
		return result
			

class MapGenerator:
	UP_RIGHT_CORNER = 302
	UP_LEFT_CORNER = 261
	DOWN_LEFT_CORNER = 271
	DOWN_RIGHT_CORNER = 311
	VERT_WALL = 241
	SQUARE = 211
	HORZ_WALL = 331

	WALL_SIZE = 1
	OUTER_WALL_SIZE = 1

	CORRIDOR_SIZE = 1

	NODE_DIST = 2

	def generatePath(self, xSize, ySize):
		graph = Graph([])

		for x in range(xSize):
			for y in range(ySize):
				newNode = (f"n{x}{y}", x, y)
				graph.addNode(newNode)

				if x > 0:
					graph.addConnection(newNode, (f"n{x-1}{y}", x-1, y), random.random())
				if y > 0:
					graph.addConnection(newNode, (f"n{x}{y-1}", x, y-1), random.random())

				if x < xSize-1:
					graph.addConnection(newNode, (f"n{x+1}{y}", x+1, y), random.random())
				if y < ySize-1:
					graph.addConnection(newNode, (f"n{x}{y+1}", x, y+1), random.random())

		mst = graph.kruskals()

		self.drawGraph(graph, mst)
		
		return mst

	def drawGraph(self, graph, highlightedEdges):
		xPoints = []
		yPoints = []

		plt.rcParams.update({'font.size': 5})

		for node in graph.nodes:
			xPoints.append(node[1])
			yPoints.append(node[2])

			plt.plot(node[1], node[2], 'ro')

			plt.text(node[1], node[2], node[0])

		for connection in highlightedEdges:
			plt.plot([connection[0][1], connection[1][1]],[connection[0][2], connection[1][2]], color='red', linewidth=1)
			plt.text(sum([connection[0][1], connection[1][1]])/2, sum([connection[0][2], connection[1][2]])/2,"{:.2f}".format(graph.connections[connection]) )
		
		plt.axis([min(xPoints) - 1, max(xPoints) + 1, min(yPoints) - 1, max(yPoints) + 1])

		plt.show()

	def generateMazeArray(self, mazeX, mazeY):
		mazePath = self.generatePath(mazeX, mazeY)

		width = 2+(self.WALL_SIZE * mazeX)+(self.CORRIDOR_SIZE*(mazeX))
		height = 2+(self.WALL_SIZE * mazeY)+(self.CORRIDOR_SIZE*(mazeY))

		wall_array = [[self.SQUARE for x in range(width)] for y in range(height)] 

		nodeDict = {}
		for edge in mazePath:
			n1 = edge[0]
			n2 = edge[1]

			if not(n1 in nodeDict.keys()):
				nodeDict[n1] = []
			if not(n2 in nodeDict.keys()):
				nodeDict[n2] = []

			if n1[1] > n2[1]:
				nodeDict[n1].append("r")
				nodeDict[n2].append("l")
			elif n1[1] < n2[1]:
				nodeDict[n1].append("l")
				nodeDict[n2].append("r")
			elif n1[2] > n2[2]:
				nodeDict[n1].append("d")
				nodeDict[n2].append("u")
			elif n1[2] < n2[2]:
				nodeDict[n1].append("u")
				nodeDict[n2].append("d")

		for node in nodeDict:
			centerX = ((node[1])*(self.CORRIDOR_SIZE+1))+self.OUTER_WALL_SIZE
			centerY = ((node[2])*(self.CORRIDOR_SIZE+1))+self.OUTER_WALL_SIZE

			right = "r" in nodeDict[node]
			left = "l" in nodeDict[node]
			up = "u" in nodeDict[node]
			down = "d" in nodeDict[node]

			wall_array[centerY][centerX] = -1
			if up:
				wall_array[centerY+1][centerX] = -1
			if down:
				wall_array[centerY-1][centerX] = -1
			if left:
				wall_array[centerY][centerX-1] = -1
			if right:
				wall_array[centerY][centerX+1] = -1

		# outside walls
		for i in range(1, width-1):
			wall_array[0][i] = self.HORZ_WALL
			wall_array[height-1][i] = self.HORZ_WALL

		for i in range(1, height-1):
			wall_array[i][0] = self.VERT_WALL
			wall_array[i][width-1] = self.VERT_WALL

		wall_array[0][0] = self.UP_LEFT_CORNER
		wall_array[0][width-1] = self.UP_RIGHT_CORNER
		wall_array[height-1][width-1] = self.DOWN_RIGHT_CORNER
		wall_array[height-1][0] = self.DOWN_LEFT_CORNER

		return wall_array

def main(argv):
	numPoints = 7

    # This parameter parsing used the follwing page for reference:
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm
	opts, args = getopt.getopt(argv,"n:")
	for opt, arg in opts:
		if opt == '-n':
			numPoints = int(arg)

	gen = MapGenerator()

	mazeArray = gen.generateMazeArray(height, width)

	xml = xml_gen.generate_xml(mazeArray)

	print(xml)

if __name__ == "__main__":
   main(sys.argv[1:])
