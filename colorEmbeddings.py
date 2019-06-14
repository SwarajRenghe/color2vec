import json
import copy
import random
import time
import numpy

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

def getClosestColor (color, allColors):
	return sorted(allColors, key=lambda x: (abs (color[0] - x["rgb"][0]) + abs (color[1] - x["rgb"][1]) + abs (color[2] - x["rgb"][2])))[0]

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
	for i in range (length):
		walk += str(getClosestColor(startColor, colors)["color"])
		walk += "/"
		startColor = variateColor (startColor)
	return walk

def colorToOneHot (color, colors):
	oneHotEncoding = [0 for i in range(0, len(colors))]
	count = 0
	for i in colors:
		if i["color"] == color:
			oneHotEncoding[count] = 1
			return oneHotEncoding
		else:
			count += 1


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
	walk = []
	for i in range (length):
		walk.append (colorToOneHot (getClosestColor(startColor, colors)["color"], colors))
		startColor = variateColor (startColor)
	return walk

wordVecSettings = {
	'window_size' : 5,
	'dimension' : 10,
	'epochs' : 50,
	'learning_rate' : 0.01
}

def createCorpusOfSentences (colors, numberOfSentences):
	training_data = []
	for i in range (numberOfSentences):
		temp = oneHotEncodingOfRandomWalk (colors, 5)
		for j, word in enumerate(temp):
			target_word = temp[j]
			context_words = []
			for k in range (j - wordVecSettings["window_size"], j+wordVecSettings["window_size"]+1):
				if k != j and k < len(temp)-1 and k >= 0:
					context_words.append (temp[k])
			training_data.append ([target_word, context_words])
	return numpy.array (training_data)

def createCorpusOfSentencesWords (colors, numberOfSentences):
	training_data = []
	for i in range (numberOfSentences):
		temp = randomWalk (colors, 5).split("/")
		# temp = oneHotEncodingOfRandomWalk (colors, 5)
		for j, word in enumerate(temp):
			target_word = temp[j]
			context_words = []
			for k in range (j - wordVecSettings["window_size"], j+wordVecSettings["window_size"]+1):
				if k != j and k < len(temp)-1 and k >= 0:
					context_words.append (temp[k])
			training_data.append ([target_word, context_words])
	return (training_data)

def softmax (x):
    e_x = numpy.exp (x - numpy.max (x))
    return e_x / e_x.sum (axis=0)

def forwardPass (target_word, weight1, weight2):
	hidden_layer = numpy.dot (weight1.T, target_word)
	secondMatrix = numpy.dot (weight2.T, hidden_layer)
	output = softmax (secondMatrix)
	return output, hidden_layer, secondMatrix

def backPropogation (error, hidden_layer, target_word, weight1, weight2):
	derivative_weight2 = numpy.outer(hidden_layer, error)
	derivative_weight1 = numpy.outer (target_word, numpy.dot (weight2, error.T))
	weight1 = weight1 - (wordVecSettings["learning_rate"] * derivative_weight1)
	weight2 = weight2 - (wordVecSettings["learning_rate"] * derivative_weight2)

def train (colors, training_data):
	weight1 = numpy.random.uniform (-1, 1, (len (colors), wordVecSettings["dimension"]))
	weight2 = numpy.random.uniform (-1, 1, (wordVecSettings["dimension"], len (colors)))

	for i in range (wordVecSettings["epochs"]):
		loss = 0
		for target_word, context_words in training_data:
			predicted_output, hidden_layer, secondMatrix = forwardPass (target_word, weight1, weight2)
			error = numpy.sum ([numpy.subtract(predicted_output, word) for word in context_words], axis=0)
			backPropogation (error, hidden_layer, target_word, weight1, weight2)
			loss += -numpy.sum ([secondMatrix[word.index(1)] for word in context_words]) + len(context_words) * numpy.log(numpy.sum(numpy.exp(secondMatrix)))
		print ("Epoch : ", i, ", Loss : ", loss)
	return weight1

def run ():
	# data cleanup and collection functions
	jsonObject = getJSON ("./color.json")
	colors = jsonObject["colors"]
	justTheColorsArray = []
	for i in colors:
		i["rgb"] = convertHexToRGB (i["hex"])
	# print (oneHotEncodingOfRandomWalk (colors, 2))
	# oneHotEncodingsOfAllColors =  generateOneHotEncodings (colors)
	training_data = createCorpusOfSentences(colors, 20)
	print (train (colors, training_data))
	# print (" ")
	# print (createCorpusOfSentences(colors, 1)[0])
	# training



run ()