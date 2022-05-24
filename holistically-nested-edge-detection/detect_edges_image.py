
import argparse
import cv2
import os
from matplotlib import pyplot as plt
import math
import numpy as np
import imutils
import convert_to_gcode

class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

class SisyphusPath:
	def __init__(self, image, num_path_points):
		self.image = image
		self.num_path_points = num_path_points

	def get_dda_ratio(self, q, r, image, intensity):
		q = [round(q[0]), round(q[1])]
		r = [round(r[0]), round(r[1])]

		whitePix = 0
		blackPix = 0

		if r[0] == q[0]:
			if r[1] < q[1]:
				for i in range(r[1], q[1]):
					if r[0] < np.ma.size(image, axis=0) - 1:
						if image[r[0], i] > intensity:
							whitePix += 1
						else:
							blackPix += 1
			else:
				for i in range(q[1], r[1]):
					if r[0] < np.ma.size(image, axis=0) - 1:
						if image[r[0], i] > intensity:
							whitePix += 1
						else:
							blackPix += 1
		else:
			slope = (r[1] - q[1])/(r[0] - q[0])

			if slope <= 1 and slope >= -1:
				if (r[0] < q[0]):
					temp = q
					q = r
					r = temp
			else:
				if (r[1] < q[1]):
					temp = q
					q = r
					r = temp

			slope = (r[1] - q[1])/(r[0] - q[0])

			if slope <= 1 and slope >= -1:
				deltaY = slope
				y = q[1]
				for x in range(q[0], r[0] + 1):
					y += deltaY
					if x < np.ma.size(image, axis=0) - 1:
						if image[x, round(y)] > intensity:
							whitePix += 1
						else:
							blackPix += 1
			else:
				deltaX = 1/slope
				x = q[0]
				for y in range(q[1], r[1] + 1):
					x += deltaX
					if x < np.ma.size(image, axis=0) - 1:
						if image[round(x), y] > intensity:
							whitePix += 1
						else:
							blackPix += 1

		if (whitePix + blackPix) == 0:
			return 1

		return whitePix/ (whitePix+blackPix)

	def get_direction(self, p1, p2):
		if (p2[0]-p1[0]) == 0:
			if p1[0] > p2[0]:
				return 270
			else:
				return 90

		slope = (p2[1]- p1[1])/(p2[0]-p1[0])
		return math.degrees(math.atan(slope))

	def getDistance(self, p1, p2):
		return math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))

	def get_weighted_score(self, p1, p2, image, intensity, lastDirection):
		distance = self.getDistance(p1, p2)

		if distance > 30:
			return [0, distance]

		direction = self.get_direction(p1, p2)
		phi = abs(lastDirection - direction) % 360
		
		angDiff = 0

		if phi > 180:
			angDiff = 360-phi
		else:
			angDiff = phi

		angDiff = angDiff/360

		# dda_ratio = get_dda_ratio(p1, p2, image, intensity)

		score = 1
		logDiff = math.log(angDiff+0.0001)*-1

		# if (dda_ratio != 0):
		# 	score = (dda_ratio**2)* (1-(distance/750))* (logDiff)
		# else:
		score = (1-(distance/750))* (logDiff)

		return [score, distance]

	def get_path(self, centers, image, intensity):
		current_path = []

		bestCenterIndex = 0
		lastDirection = 0

		total = 0
		defaults = 0

		currentCenter = centers[0]

		while len(centers) > 0:
			bestCenterIndex = 0
			leastDistance = 10000000
			leastDistanceIndex = 0
			greatestScore = 0
			for i in range(len(centers)):
				score, distance = self.get_weighted_score(currentCenter, centers[i], image, intensity, lastDirection)
				if score > greatestScore:
					greatestScore = score
					bestCenterIndex = i
				
				if distance < leastDistance:
					leastDistance = distance
					leastDistanceIndex = i

			if greatestScore == 0:
				defaults += 1
				bestCenterIndex = leastDistanceIndex

			total += 1
			bestCenter = centers[bestCenterIndex]
			lastDirection = self.get_direction(currentCenter, bestCenter)

			current_path.append(currentCenter)

			currentCenter = centers.pop(bestCenterIndex)

			if len(centers) == 0:
				current_path.append(currentCenter)
				current_path.append(current_path[0])

		print(f"total: {total}, defaults: {defaults}")
		return current_path

	def rotateArray(self, arr, n):
		for i in range(n):
			arr.append(arr.pop(0))
		return arr

	def mergePaths(self, path1, path2):
		if len(path1) == 0:
			return path2
		if len(path2) == 0:
			return path1

		smallestDistance = 100000
		bestp1 = 0
		bestp2 = 0
		for i in range(len(path1)):
			for j in range(len(path2)):
				dist = self.getDistance(path1[i], path2[j])
				if dist < smallestDistance:
					smallestDistance = dist
					bestp1 = i
					bestp2 = j

		path1 = self.rotateArray(path1, bestp1)
		path2 = self.rotateArray(path2, bestp2)

		finalPath = [path1[0]]
		finalPath.extend(path2)
		finalPath.extend([path2[0]])
		finalPath.extend([path1[0]])
		finalPath.extend(path1[1:])

		return finalPath

	def getFullPath(self, paths, cnts):
		newPath = paths.pop(0)
		while len(paths) > 0:
			finalPoint = newPath[-1]

			leastDistance = 100000
			bestGroup = 0
			for i in range(len(paths)):
				for j in range(len(paths[i])):
					dist = self.getDistance(finalPoint, paths[i][j])
					if dist < leastDistance:
						leastDistance = dist
						bestGroup = i

			newPath = self.mergePaths(newPath, paths.pop(bestGroup))

		newPath.append(newPath[0])
		return newPath

	def getCentroidGroups(self, image, centroids, intensity):
		# find all the 'white' shapes in the image
		lower = np.array([intensity])
		upper = np.array([255])
		shapeMask = cv2.inRange(image, lower, upper)

		# find the contours in the mask
		cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_LIST,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		print("I found {} white shapes".format(len(cnts)))

		img = cv2.drawContours(image, cnts, -1, (0, 255, 0), 1)

		for contour in cnts:
			cv2.fillPoly(img, pts =contour, color=(255,255,255))

		# plt.imshow(img, interpolation='nearest', aspect='auto')

		# plt.show()

		# cv2.waitKey(0)
		
		cnts.reverse()

		groups = [[] for i in range(len(cnts))]

		rejectGroup = []

		for centroid in centroids:
			candidates = []
			for i in range(len(cnts)):
				result = cv2.pointPolygonTest(cnts[i], (centroid[0], centroid[1]), False) 
				dist = cv2.pointPolygonTest(cnts[i], (centroid[0], centroid[1]), True)
				if result == 1 or result == 0:
					candidates.append([i, dist])
			
			if len(candidates) == 0:
				continue

			primeCandiate = candidates[0]
			for cand in candidates:
				if cand[1] < primeCandiate[1]:
					primeCandiate = cand

			groups[primeCandiate[0]].append(centroid)

		groups = [i for i in groups if len(i) >0]

		currentColor = 20
		for cnt in cnts:
			cv2.fillPoly(image, pts =[cnt], color=(currentColor))
			currentColor = (currentColor + 100)%255

		return (groups, cnts)

plt.interactive(False)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True,
	help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("--canny", type=bool, required=False, default=False)
ap.add_argument("--centroids", type=int, required=False, default=100)
args = vars(ap.parse_args())

k_means_pixels = []

chosenimage = []

image = cv2.imread(args["image"])

if args["canny"]:
	print("Running Canny...")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	canny = cv2.Canny(blurred, 30, 150)
	chosenimage = canny
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	chosenimage = cv2.dilate(chosenimage, kernel, iterations=1)
else:
	print("Running HED...")
	protoPath = os.path.sep.join([args["edge_detector"],
		"deploy.prototxt"])
	modelPath = os.path.sep.join([args["edge_detector"],
		"hed_pretrained_bsds.caffemodel"])
	net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	cv2.dnn_registerLayer("Crop", CropLayer)
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
		mean=(104.00698793, 116.66876762, 122.67891434),
		swapRB=False, crop=False)
	net.setInput(blob)
	hed = net.forward()
	hed = cv2.resize(hed[0, 0], (W, H))
	hed = (255 * hed).astype("uint8")
	chosenimage = hed

intensity = 30

for i in range(len(chosenimage)):
	for j in range(len(chosenimage[i])):
		if chosenimage[i][j] > intensity:
			new_row = [j,i]
			k_means_pixels.append(new_row)

pixels32 = np.float32(k_means_pixels)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
ret,label,center=cv2.kmeans(pixels32,args["centroids"],None,criteria,10,cv2.KMEANS_PP_CENTERS)

centerList = center.tolist()

groups, cnts = getCentroidGroups(chosenimage.copy(), centerList, intensity)

paths = []
for group in groups:
	path_matrix = get_path(group.copy(), chosenimage, intensity)
	paths.append(path_matrix)

finalPath = getFullPath(paths, cnts)

plt.imshow(image, interpolation='nearest', aspect='auto')

xPath = []
yPath = []
for path in finalPath:
	xPath.append(path[0])
	yPath.append(path[1])

plt.plot(xPath,yPath, color='red', linewidth=1)

for group in groups:
	group = np.array(group)
	plt.scatter(group[:,0],group[:,1],s = 20, marker = 's')

plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()

cv2.waitKey(0)

zippedPath = zip(xPath, yPath)

gcode = convert_to_gcode.convertToGcode(zippedPath, len(chosenimage), len(chosenimage[i]))

gcodeFile = open("./gcodes/newCode.gcode", "w")
gcodeFile.writelines(gcode)
gcodeFile.close()
