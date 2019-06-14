import json
import copy
import random
import time

def getJSON (filename):
	with open(filename, "r") as read_file:
	    data = json.load(read_file)
	return data

def convertHexToRGB (colorInHex):
	colorInHex = colorInHex.lstrip ('#')
	return tuple (int (colorInHex[i:i+2], 16) for i in (0, 2, 4))

def getSortedListOfClosestColors (color, allColors):
	"""
		colorInRGB - a tuple
		allColors - a list of dictionaries
	"""
	return sorted(allColors, key=lambda x: (abs (color[0] - x["rgb"][0]) + abs (color[1] - x["rgb"][1]) + abs (color[2] - x["rgb"][2])))

def constrainValue (variable, lower_limit, upper_limit, type):
	if type == "upper_limit":
		if variable < lower_limit:
			return upper_limit
		if variable > upper_limit:
			return upper_limit
		return variable
	elif type == "lower_limit":
		if variable < lower_limit:
			return lower_limit
		if variable > upper_limit:
			return lower_limit
		return variable
	elif type == "middle":
		if variable < lower_limit:
			return (lower_limit + upper_limit) / 2
		if variable > upper_limit:
			return (lower_limit + upper_limit) / 2
		return variable
	else:
		return variable

def variateColor (color):
	a = random.randint (-50, 50)
	b = random.randint (-50, 50)
	c = random.randint (-50, 50)

	a = constrainValue (color[0] + a, 0, 255, "upper_limit")
	b = constrainValue (color[1] + b, 0, 255, "upper_limit")
	c = constrainValue (color[2] + c, 0, 255, "upper_limit")

	return (a, b, c)

def randomWalk (colors, length):
	startColor = (random.randint (0, 255), random.randint (0, 255), random.randint (0, 255))
	walk = ""
	for i in range (10):
		temp = getSortedListOfClosestColors (startColor, colors)
		walk += str(temp[0]["color"])
		walk += " "
		startColor = variateColor (startColor)
	return walk

def colorToOneHot (color, colors):
	oneHotEncoding = [0 for i in range(0, len(colors))]

def generateOneHotEncodings (colors):
	numberOfColors = len(colors)
	oneHotEncodings = []
	count = 0
	for i in colors:
		color = [0 for i in range(0, numberOfColors)]
		color[count] = 1
		count += 1
		oneHotEncodings.append (color)
	return oneHotEncodings


def oneHotEncodingOfRandomWalk (colors, length):
	startColor = (random.randint (0, 255), random.randint (0, 255), random.randint (0, 255))


wordVecSettings = {
	'window_size' : 2,
	'dimension' : 10,
	'epochs' : 50,
	'learning_rate' : 0.01
}

def run ():
	# data cleanup and collection functions
	jsonObject = getJSON ("./color.json")
	colors = jsonObject["colors"]
	justTheColorsArray = []
	for i in colors:
		i["rgb"] = convertHexToRGB (i["hex"])
	print (randomWalk (colors, 10))
	print (generateOneHotEncodings (colors)[1])

	# word vec building functions

run ()