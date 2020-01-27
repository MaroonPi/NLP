import os
import sys
import glob
import random

#Prior probabilities
priorPositive = 0.5
priorNegative = 0.5
priorTrue = 0.5
priorFalse = 0.5

#Getting the dictionaries with conditional probabilities from nbmodel.txt
dicts_from_file = []
with open('nbmodel.txt','r') as inf:
    for line in inf:
        dicts_from_file.append(eval(line))

positiveProbDict = dicts_from_file[0]
negativeProbDict = dicts_from_file[1]
trueProbDict = dicts_from_file[2]
falseProbDict = dicts_from_file[3]

#Defining list of punctuation marks
punctuation = [',','.','?',';','/','*',':','-','(',')','"','','&','@','$','+','%','\\','#','=']

#Defining list of stop words
stopWords = ['to','from','or','and','the','an','of','a','for','is','as','in','are','on','was','were','has','have','had','been','be','what',
'where','why','when','which','how','who','whom','here','there','this','that','these','those','i','you','we','he','she','they','it','them','our','ours','your','yours',
'i\'m','am','my','me','his','her','their','they\'re','at','us','by','i\'ve','they\'ve','we\'ve','you\'ve','']

#Getting the test files, given the root of test data.
testFiles = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

#Creating the output file
outputFile = open("nboutput.txt","w")

#Reading each file in test data and assigning labels
for f in testFiles:
    #Reading the text from the test file
    testFile = open(f,"r")
    testContent = testFile.readlines()
    testContent = [x.strip() for x in testContent]
    testString = str(testContent[0])

    #Remove punctuation marks
    testText = ""
    for i in testString:
        if i not in punctuation:
            testText += i

    #Convert to lower case
    testText = testText.lower()

    #Creating list of test words
    testWords = testText.split(" ")

    #Removing white space in list of test words
    for i in testWords:
        i = i.strip()

    #Removing filler words in test word list
    for i in range(len(testWords)-1,-1,-1):
        if(testWords[i] in stopWords):
            testWords.remove(testWords[i])
        elif(testWords[i][0].isdigit()):
            testWords.remove(testWords[i])

    #Collecting all the test words in the file into a dictionary
    testDict = {}
    for i in testWords:
        if i not in testDict:
            testDict[i] = 1
        else:
            testDict[i] += 1

    #Calculating positive conditional probability
    pPositiveConditional = priorPositive
    for i in testDict:
        #If the word is in positive dictionary, otherwise ignoring
        if i in positiveProbDict:
            pPositiveConditional *= float(positiveProbDict[i])
            pPositiveConditional = float('{:.350f}'.format(pPositiveConditional))

    #Calculating negative conditional probability
    pNegativeConditional = priorNegative
    for i in testDict:
        #If the word is in negative dictionary, otherwise ignoring
        if i in negativeProbDict:
            pNegativeConditional *= float(negativeProbDict[i])
            pNegativeConditional = float('{:.350f}'.format(pNegativeConditional))

    #Calculating true conditional probability
    pTrueConditional = priorTrue
    for i in testDict:
        #If the word is in true dictionary, otherwise ignoring
        if i in trueProbDict:
            pTrueConditional *= float(trueProbDict[i])
            pTrueConditional = float('{:.350f}'.format(pTrueConditional))

    #Calculating false conditional probability
    pFalseConditional = priorFalse
    for i in testDict:
        #If the word is in false dictionary, otherwise ignoring
        if i in falseProbDict:
            pFalseConditional *= float(falseProbDict[i])
            pFalseConditional = float('{:.350f}'.format(pFalseConditional))

    #Writing the labels to output file
    if(pTrueConditional>pFalseConditional):
        outputFile.write("truthful ")
    elif(pTrueConditional<pFalseConditional):
        outputFile.write("deceptive ")
    else:
        outputFile.write(random.choice(["deceptive ","truthful "]))

    if(pPositiveConditional>pNegativeConditional):
        outputFile.write("positive ")
    elif(pPositiveConditional<pNegativeConditional):
        outputFile.write("negative ")
    else:
        outputFile.write(random.choice(["negative ","positive "]))
    outputFile.write(str(f)+"\n")
