import os
import sys
import glob

#Getting the lists from the model
lists_from_file = []
modelFile = sys.argv[1]
with open(modelFile,'r') as f:
    for line in f:
        lists_from_file.append(eval(line))

features = lists_from_file[0]
posNegWeights = lists_from_file[1][0]
posNegBias = lists_from_file[1][1]
trueFalseWeights = lists_from_file[2][0]
trueFalseBias = lists_from_file[2][1]

#Defining list of punctuation marks
punctuation = [',','.','?','!',';','/','*',':','-','(',')','"','','&','@','$','+','%','\\','#','=','[',']','<','>','~']

#Defining list of stop words
stopWords = ['to','from','or','and','the','an','of','a','for','is','as','in','are','on','was','were','has','have','had','been','be','what',
'where','why','when','which','how','who','whom','here','there','this','that','these','those','i','you','we','he','she','they','it','them','our','ours','your','yours',
'i\'m','am','my','me','his','her','their','they\'re','at','us','by','i\'ve','they\'ve','we\'ve','you\'ve','','i\'d','we\'d','she\'d','he\'d','they\'d','you\'d','it\'d',
'therefore','will','shall','per','than','because','with','besides','i\'ll','you\'ll','he\'ll','she\'ll','they\'ll','it\'ll','him','her','did','he\'s','she\'s','it\'s','we\'ll','its']

#Getting the test files, given the root of test data.
testFiles = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))

#Creating the output file
outputFile = open("percepoutput.txt","w")

#Reading each file in test data and assigning labels
for f in testFiles:
    #Processing the file
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

    #Creating list of words in the file
    testWords = testText.split(" ")

    #Removing white space and quotes in list of words in the file
    for i in testWords:
        i = i.strip()
        i = i.strip('\'')
        i = i.lstrip('\'')
        i = i.rstrip('\'')

    #Removing filler words in list of words in the file
    for i in range(len(testWords)-1,-1,-1):
        if(testWords[i] in stopWords or (len(testWords[i])<=2 and not(testWords[i]=='no' or testWords[i]=='ok'))):
            testWords.remove(testWords[i])
        elif(not(testWords[i][0].isalpha())):
            testWords.remove(testWords[i])

    #For every word in the list of words
    testInput = [0] * len(features)
    for i in testWords:
        if i in features:
            reqIndex = features.index(i)
            testInput[reqIndex] += 1

    #Running classifier for truthful-deceptive
    trueFalseActivation = 0
    for i in range(len(features)):
        trueFalseActivation += (testInput[i]*trueFalseWeights[i])
    trueFalseActivation += trueFalseBias
    if(trueFalseActivation<=0):
        outputFile.write("deceptive ")
    else:
        outputFile.write("truthful ")

    #Running classifier for positive-negative
    posNegActivation = 0
    for i in range(len(features)):
        posNegActivation += (testInput[i]*posNegWeights[i])
    posNegActivation += posNegBias
    if(posNegActivation<=0):
        outputFile.write("negative ")
    else:
        outputFile.write("positive ")
    outputFile.write(str(f)+"\n")
