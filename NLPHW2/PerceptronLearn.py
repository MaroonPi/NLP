import os
import sys
import glob
import operator

#List all the files, given the root of training data
all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

#Training data
train_by_class = {}
train_by_class["negative_polarity truthful_from_Web"] = []
train_by_class["negative_polarity deceptive_from_MTurk"] = []
train_by_class["positive_polarity truthful_from_TripAdvisor"] = []
train_by_class["positive_polarity deceptive_from_MTurk"] = []

#Splitting files into the train set
for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    train_by_class[class1+" "+class2].append(f)

#Creating dictionaries for words of each class
positiveDict = {}
negativeDict = {}
trueDict = {}
falseDict = {}

#Creating dictionary for entire set of words
vocabularyDict = {}

#Defining list of punctuation marks
punctuation = [',','.','?','!',';','/','*',':','-','(',')','"','','&','@','$','+','%','\\','#','=','[',']','<','>','~']

#Defining list of stop words
stopWords = ['to','from','or','and','the','an','of','a','for','is','as','in','are','on','was','were','has','have','had','been','be','what',
'where','why','when','which','how','who','whom','here','there','this','that','these','those','i','you','we','he','she','they','it','them','our','ours','your','yours',
'i\'m','am','my','me','his','her','their','they\'re','at','us','by','i\'ve','they\'ve','we\'ve','you\'ve','','i\'d','we\'d','she\'d','he\'d','they\'d','you\'d','it\'d',
'therefore','will','shall','per','than','because','with','besides','i\'ll','you\'ll','he\'ll','she\'ll','they\'ll','it\'ll','him','her','did','he\'s','she\'s','it\'s','we\'ll','its']

posNegFileDict = {}    #Holding files classifying them as positive(+1) or negative(-1) unordered
trueFalseFileDict = {}   #Holding files classifying them as true(+1) or false(-1) unordered

#For each true positive file in train set
for f in train_by_class["positive_polarity truthful_from_TripAdvisor"]:
    posNegFileDict[f] = 1
    trueFalseFileDict[f] = 1
    truePositiveFile = open(f,"r")
    truePositiveContent = truePositiveFile.readlines()
    truePositiveContent = [x.strip() for x in truePositiveContent]
    truePositiveString = str(truePositiveContent[0])

    #Remove punctuation marks
    truePositiveText = ""
    for i in truePositiveString:
        if i not in punctuation:
            truePositiveText += i

    #Convert to lower case
    truePositiveText = truePositiveText.lower()

    #Creating list of true positive words
    truePositiveWords = truePositiveText.split(" ")

    #Removing white space and quotes in list of true positive words
    for i in truePositiveWords:
        i = i.strip()
        i = i.strip('\'')
        i = i.lstrip('\'')
        i = i.rstrip('\'')

    #Removing filler words in true positive list
    for i in range(len(truePositiveWords)-1,-1,-1):
        if(truePositiveWords[i] in stopWords or (len(truePositiveWords[i])<=2 and not(truePositiveWords[i]=='no' or truePositiveWords[i]=='ok'))):
            truePositiveWords.remove(truePositiveWords[i])
        elif(not(truePositiveWords[i][0].isalpha())):
            truePositiveWords.remove(truePositiveWords[i])

    for i in truePositiveWords:
        if i not in positiveDict:
            positiveDict[i] = 1
        else:
            positiveDict[i] += 1
        if i not in trueDict:
            trueDict[i] = 1
        else:
            trueDict[i] += 1
        if i not in vocabularyDict:
            vocabularyDict[i] = 1
        else:
            vocabularyDict[i] += 1

#For each false positive file in train set
for f in train_by_class["positive_polarity deceptive_from_MTurk"]:
    posNegFileDict[f] = 1
    trueFalseFileDict[f] = -1
    falsePositiveFile = open(f,"r")
    falsePositiveContent = falsePositiveFile.readlines()
    falsePositiveContent = [x.strip() for x in falsePositiveContent]
    falsePositiveString = str(falsePositiveContent[0])

    #Remove punctuation marks
    falsePositiveText = ""
    for i in falsePositiveString:
        if i not in punctuation:
            falsePositiveText += i

    #Convert to lower case
    falsePositiveText = falsePositiveText.lower()

    #Creating list of false positive words
    falsePositiveWords = falsePositiveText.split(" ")

    #Removing white space and quotes in list of false positive words
    for i in falsePositiveWords:
        i = i.strip()
        i = i.strip('\'')
        i = i.lstrip('\'')
        i = i.rstrip('\'')

    #Removing filler words in false positive list
    for i in range(len(falsePositiveWords)-1,-1,-1):
        if(falsePositiveWords[i] in stopWords or (len(falsePositiveWords[i])<=2 and not(falsePositiveWords[i]=='no' or falsePositiveWords[i]=='ok'))):
            falsePositiveWords.remove(falsePositiveWords[i])
        elif(not(falsePositiveWords[i][0].isalpha())):
            falsePositiveWords.remove(falsePositiveWords[i])

    for i in falsePositiveWords:
        if i not in positiveDict:
            positiveDict[i] = 1
        else:
            positiveDict[i] += 1
        if i not in falseDict:
            falseDict[i] = 1
        else:
            falseDict[i] += 1
        if i not in vocabularyDict:
            vocabularyDict[i] = 1
        else:
            vocabularyDict[i] += 1

#For each true negative file in train set
for f in train_by_class["negative_polarity truthful_from_Web"]:
    posNegFileDict[f] = -1
    trueFalseFileDict[f] = 1
    trueNegativeFile = open(f,"r")
    trueNegativeContent = trueNegativeFile.readlines()
    trueNegativeContent = [x.strip() for x in trueNegativeContent]
    trueNegativeString = str(trueNegativeContent[0])

    #Remove punctuation marks
    trueNegativeText = ""
    for i in trueNegativeString:
        if i not in punctuation:
            trueNegativeText += i

    #Convert to lower case
    trueNegativeText = trueNegativeText.lower()

    #Creating list of true negative words
    trueNegativeWords = trueNegativeText.split(" ")

    #Removing white space and quotes in list of true negative words
    for i in trueNegativeWords:
        i = i.strip()
        i = i.strip('\'')
        i = i.lstrip('\'')
        i = i.rstrip('\'')

    #Removing filler words in true negative list
    for i in range(len(trueNegativeWords)-1,-1,-1):
        if(trueNegativeWords[i] in stopWords or (len(trueNegativeWords[i])<=2 and not(trueNegativeWords[i]=='no' or trueNegativeWords[i]=='ok'))):
            trueNegativeWords.remove(trueNegativeWords[i])
        elif(not(trueNegativeWords[i][0].isalpha())):
            trueNegativeWords.remove(trueNegativeWords[i])

    for i in trueNegativeWords:
        if i not in trueDict:
            trueDict[i] = 1
        else:
            trueDict[i] += 1
        if i not in negativeDict:
            negativeDict[i] = 1
        else:
            negativeDict[i] += 1
        if i not in vocabularyDict:
            vocabularyDict[i] = 1
        else:
            vocabularyDict[i] += 1

#For each false negative file in train set
for f in train_by_class["negative_polarity deceptive_from_MTurk"]:
    posNegFileDict[f] = -1
    trueFalseFileDict[f] = -1
    falseNegativeFile = open(f,"r")
    falseNegativeContent = falseNegativeFile.readlines()
    falseNegativeContent = [x.strip() for x in falseNegativeContent]
    falseNegativeString = str(falseNegativeContent[0])

    #Remove punctuation marks
    falseNegativeText = ""
    for i in falseNegativeString:
        if i not in punctuation:
            falseNegativeText += i

    #Convert to lower case
    falseNegativeText = falseNegativeText.lower()

    #Creating list of false negative words
    falseNegativeWords = falseNegativeText.split(" ")

    #Removing white space and quotes in list of false negative words
    for i in falseNegativeWords:
        i = i.strip()
        i = i.strip('\'')
        i = i.lstrip('\'')
        i = i.rstrip('\'')

    #Removing filler words in false negative list
    for i in range(len(falseNegativeWords)-1,-1,-1):
        if(falseNegativeWords[i] in stopWords or (len(falseNegativeWords[i])<=2 and not(falseNegativeWords[i]=='no' or falseNegativeWords[i]=='ok'))):
            falseNegativeWords.remove(falseNegativeWords[i])
        elif(not(falseNegativeWords[i][0].isalpha())):
            falseNegativeWords.remove(falseNegativeWords[i])

    for i in falseNegativeWords:
        if i not in falseDict:
            falseDict[i] = 1
        else:
            falseDict[i] += 1
        if i not in negativeDict:
            negativeDict[i] = 1
        else:
            negativeDict[i] += 1
        if i not in vocabularyDict:
            vocabularyDict[i] = 1
        else:
            vocabularyDict[i] += 1

#Sorting vocabularyDict by frequency of words
sorted_Vocabulary = sorted(vocabularyDict.items(),key=operator.itemgetter(1),reverse=True)

#Getting the 1000 most common words
features = []
total = 0
for i in range(len(sorted_Vocabulary)):
    features.append(sorted_Vocabulary[i][0])
    total += 1
    if(total==1000):
        break

#Creating input and output arrays
inputArray = []
posNegOutputArray = []
trueFalseOutputArray = []

for f in all_files:
    #Assigning correct output of file
    posNegOutputArray.append(posNegFileDict[f])
    trueFalseOutputArray.append(trueFalseFileDict[f])

    #Processing the file
    inputFile = open(f,"r")
    inputContent = inputFile.readlines()
    inputContent = [x.strip() for x in inputContent]
    inputString = str(inputContent[0])

    #Remove punctuation marks
    inputText = ""
    for i in inputString:
        if i not in punctuation:
            inputText += i

    #Convert to lower case
    inputText = inputText.lower()

    #Creating list of words in the file
    inputWords = inputText.split(" ")

    #Removing white space and quotes in list of words in the file
    for i in inputWords:
        i = i.strip()
        i = i.strip('\'')
        i = i.lstrip('\'')
        i = i.rstrip('\'')

    #Removing filler words in list of words in the file
    for i in range(len(inputWords)-1,-1,-1):
        if(inputWords[i] in stopWords or (len(inputWords[i])<=2 and not(inputWords[i]=='no' or inputWords[i]=='ok'))):
            inputWords.remove(inputWords[i])
        elif(not(inputWords[i][0].isalpha())):
            inputWords.remove(inputWords[i])

    #For every word in the list of words
    countWords = [0] * len(features)
    for i in inputWords:
        if i in features:
            reqIndex = features.index(i)
            countWords[reqIndex] += 1

    #Add it to the input
    inputArray.append(countWords)

#Setting number of iterations
numVanillaIterations = 80
numAveragedIterations = 80

#Creating the output for the vanilla perceptron
vanillaOutput = []
vanillaOutput.append(features)

#Vanilla perceptron for positive-negative classifier
#Initializing weights and bias
posNegVanillaWeights = [0] * 1000
posNegVanillaBias = 0

#Running perceptron for set amount of iterations
for i in range(numVanillaIterations):
    #Calculating activation for every example
    for j in range(len(inputArray)):
        posNegVanillaActivation = 0
        #For every feature
        for k in range(1000):
            posNegVanillaActivation += (inputArray[j][k] * posNegVanillaWeights[k])
        posNegVanillaActivation += posNegVanillaBias
        #Updating
        if((posNegOutputArray[j]*posNegVanillaActivation)<=0):
            #Update every weight
            for k in range(1000):
                posNegVanillaWeights[k] = posNegVanillaWeights[k] + (posNegOutputArray[j]*inputArray[j][k])
            posNegVanillaBias = posNegVanillaBias + posNegOutputArray[j]

posNegVanilla = [posNegVanillaWeights,posNegVanillaBias]
vanillaOutput.append(posNegVanilla)

#Vanilla perceptron for truthful-deceptive classifier
#Initializing weights and bias
trueFalseVanillaWeights = [0] * 1000
trueFalseVanillaBias = 0

#Running perceptron for set amount of iterations
for i in range(numVanillaIterations):
    #Calculating activation for every example
    for j in range(len(inputArray)):
        trueFalseVanillaActivation = 0
        #For every feature
        for k in range(1000):
            trueFalseVanillaActivation += (inputArray[j][k] * trueFalseVanillaWeights[k])
        trueFalseVanillaActivation += trueFalseVanillaBias
        #Updating
        if((trueFalseOutputArray[j]*trueFalseVanillaActivation)<=0):
            #Update every weight
            for k in range(1000):
                trueFalseVanillaWeights[k] = trueFalseVanillaWeights[k] + (trueFalseOutputArray[j]*inputArray[j][k])
            trueFalseVanillaBias = trueFalseVanillaBias + trueFalseOutputArray[j]

trueFalseVanilla = [trueFalseVanillaWeights,trueFalseVanillaBias]
vanillaOutput.append(trueFalseVanilla)

#Writing vanilla perceptron outputs to file
with open('vanillamodel.txt', 'w') as f:
    for item in vanillaOutput:
        f.write("%s\n" % item)

#Creating output for averaged perceptron
averagedOutput = []
averagedOutput.append(features)

#Averaged perceptron for positive-negative classifier
#Initializing weights and bias
posNegAveragedWeights = [0] * 1000
posNegAveragedBias = 0
posNegAveragedCachedWeights = [0] * 1000
posNegAveragedCachedBias = 0
posNegCounter = 1
for i in range(numAveragedIterations):
    #For every example
    for j in range(len(inputArray)):
        posNegAveragedActivation = 0
        #For every feature
        for k in range(1000):
            posNegAveragedActivation += (inputArray[j][k]*posNegAveragedWeights[k])
        posNegAveragedActivation += posNegAveragedBias
        #Updating
        if((posNegAveragedActivation*posNegOutputArray[j])<=0):
            #Update weights and bias
            for k in range(1000):
                posNegAveragedWeights[k] = posNegAveragedWeights[k] + (posNegOutputArray[j]*inputArray[j][k])
            posNegAveragedBias = posNegAveragedBias + posNegOutputArray[j]
            #Updated cached weights and cached bias
            for k in range(1000):
                posNegAveragedCachedWeights[k] = posNegAveragedCachedWeights[k] + (posNegOutputArray[j]*posNegCounter*inputArray[j][k])
            posNegAveragedCachedBias = posNegAveragedCachedBias + (posNegOutputArray[j]*posNegCounter)
        #Incrementing counter
        posNegCounter += 1

posNegAveragedCachedWeights = [int(round(float(i)/float(posNegCounter))) for i in posNegAveragedCachedWeights]
posNegFinalAveragedWeights = [a-b for a,b in zip(posNegAveragedWeights,posNegAveragedCachedWeights)]
posNegFinalAveragedBias = posNegAveragedBias - (int(round(float(posNegAveragedCachedBias)/float(posNegCounter))))
posNegAveraged = [posNegFinalAveragedWeights,posNegFinalAveragedBias]
averagedOutput.append(posNegAveraged)

#Averaged perceptron for truthful-deceptive classifier
#Initializing weights and bias
trueFalseAveragedWeights = [0] * 1000
trueFalseAveragedBias = 0
trueFalseAveragedCachedWeights = [0] * 1000
trueFalseAveragedCachedBias = 0
trueFalseCounter = 1
for i in range(numAveragedIterations):
    #For every example
    for j in range(len(inputArray)):
        trueFalseAveragedActivation = 0
        #For every feature
        for k in range(1000):
            trueFalseAveragedActivation += (inputArray[j][k]*trueFalseAveragedWeights[k])
        trueFalseAveragedActivation += trueFalseAveragedBias
        #Updating
        if((trueFalseAveragedActivation*trueFalseOutputArray[j])<=0):
            #Update weights and bias
            for k in range(1000):
                trueFalseAveragedWeights[k] = trueFalseAveragedWeights[k] + (trueFalseOutputArray[j]*inputArray[j][k])
            trueFalseAveragedBias = trueFalseAveragedBias + trueFalseOutputArray[j]
            #Updated cached weights and cached bias
            for k in range(1000):
                trueFalseAveragedCachedWeights[k] = trueFalseAveragedCachedWeights[k] + (trueFalseOutputArray[j]*trueFalseCounter*inputArray[j][k])
            trueFalseAveragedCachedBias = trueFalseAveragedCachedBias + (trueFalseOutputArray[j]*trueFalseCounter)
        #Incrementing counter
        trueFalseCounter += 1

trueFalseAveragedCachedWeights = [int(round(float(i)/float(trueFalseCounter))) for i in trueFalseAveragedCachedWeights]
trueFalseFinalAveragedWeights = [a-b for a,b in zip(trueFalseAveragedWeights,trueFalseAveragedCachedWeights)]
trueFalseFinalAveragedBias = trueFalseAveragedBias - (int(round(float(trueFalseAveragedCachedBias)/float(trueFalseCounter))))
trueFalseAveraged = [trueFalseFinalAveragedWeights,trueFalseFinalAveragedBias]
averagedOutput.append(trueFalseAveraged)

#Writing averaged perceptron outputs to file
with open('averagedmodel.txt', 'w') as f:
    for item in averagedOutput:
        f.write("%s\n" % item)
