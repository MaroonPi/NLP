import os
import sys
import glob

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
punctuation = [',','.','?',';','/','*',':','-','(',')','"','','&','@','$','+','%','\\','#','=']

#Defining list of stop words
stopWords = ['to','from','or','and','the','an','of','a','for','is','as','in','are','on','was','were','has','have','had','been','be','what',
'where','why','when','which','how','who','whom','here','there','this','that','these','those','i','you','we','he','she','they','it','them','our','ours','your','yours',
'i\'m','am','my','me','his','her','their','they\'re','at','us','by','i\'ve','they\'ve','we\'ve','you\'ve','']

#For each true positive file in train set
for f in train_by_class["positive_polarity truthful_from_TripAdvisor"]:
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

    #Removing white space in list of true positive words
    for i in truePositiveWords:
        i = i.strip()

    #Removing filler words in true positive list
    for i in range(len(truePositiveWords)-1,-1,-1):
        if(truePositiveWords[i] in stopWords):
            truePositiveWords.remove(truePositiveWords[i])
        elif(truePositiveWords[i][0].isdigit()):
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

    #Removing white space in list of false positive words
    for i in falsePositiveWords:
        i = i.strip()

    #Removing filler words in false positive list
    for i in range(len(falsePositiveWords)-1,-1,-1):
        if(falsePositiveWords[i] in stopWords):
            falsePositiveWords.remove(falsePositiveWords[i])
        elif(falsePositiveWords[i][0].isdigit()):
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

    #Removing white space in list of true negative words
    for i in trueNegativeWords:
        i = i.strip()

    #Removing filler words in true negative list
    for i in range(len(trueNegativeWords)-1,-1,-1):
        if(trueNegativeWords[i] in stopWords):
            trueNegativeWords.remove(trueNegativeWords[i])
        elif(trueNegativeWords[i][0].isdigit()):
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

    #Removing white space in list of false negative words
    for i in falseNegativeWords:
        i = i.strip()

    #Removing filler words in false negative list
    for i in range(len(falseNegativeWords)-1,-1,-1):
        if(falseNegativeWords[i] in stopWords):
            falseNegativeWords.remove(falseNegativeWords[i])
        elif(falseNegativeWords[i][0].isdigit()):
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

#Counting the total number of words in each dictionary
numPositiveWords = sum(positiveDict.values())
numNegativeWords = sum(negativeDict.values())
numTrueWords = sum(trueDict.values())
numFalseWords = sum(falseDict.values())

#Counting total distinct words in vocabulary
vocabulary = len(vocabularyDict)

#Applying smoothing for positive words
probPositiveDict = {}
for i in vocabularyDict:
    conditionalPositive = 1.0
    if i not in positiveDict:
        conditionalPositive = float(1)/float(numPositiveWords+vocabulary)
    else:
        conditionalPositive = float(positiveDict[i]+1)/float(numPositiveWords+vocabulary)
    conditionalPositive = '{:.50f}'.format(conditionalPositive)
    probPositiveDict[i] = conditionalPositive

#Applying smoothing for negative words
probNegativeDict = {}
for i in vocabularyDict:
    conditionalNegative = 1.0
    if i not in negativeDict:
        conditionalNegative = float(1)/float(numNegativeWords+vocabulary)
    else:
        conditionalNegative = float(negativeDict[i]+1)/float(numNegativeWords+vocabulary)
    conditionalNegative = '{:.50f}'.format(conditionalNegative)
    probNegativeDict[i] = conditionalNegative

#Applying smoothing for true words
probTrueDict = {}
for i in vocabularyDict:
    conditionalTrue = 1.0
    if i not in trueDict:
        conditionalTrue = float(1)/float(numTrueWords+vocabulary)
    else:
        conditionalTrue = float(trueDict[i]+1)/float(numTrueWords+vocabulary)
    conditionalTrue = '{:.50f}'.format(conditionalTrue)
    probTrueDict[i] = conditionalTrue

#Applying smoothing for false words
probFalseDict = {}
for i in vocabularyDict:
    conditionalFalse = 1.0
    if i not in falseDict:
        conditionalFalse = float(1)/float(numFalseWords+vocabulary)
    else:
        conditionalFalse = float(falseDict[i]+1)/float(numFalseWords+vocabulary)
    conditionalFalse = '{:.50f}'.format(conditionalFalse)
    probFalseDict[i] = conditionalFalse

#Writing all the dictionaries with probabilities to nbmodel.txt
outputFile = open("nbmodel.txt","w")
outputFile.write(str(probPositiveDict)+"\n")
outputFile.write(str(probNegativeDict)+"\n")
outputFile.write(str(probTrueDict)+"\n")
outputFile.write(str(probFalseDict)+"\n")
