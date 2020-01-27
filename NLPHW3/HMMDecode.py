import sys
import operator
import random

#Get the dictionaries from hmmmodel.txt file
dicts_from_file = []
with open('hmmmodel.txt','r',encoding="utf-8") as inf:
    for line in inf:
        dicts_from_file.append(eval(line))

transitionProbDict = dicts_from_file[0]
emissionProbDict = dicts_from_file[1]
tagDict = dicts_from_file[2]

#Creating the output file
outputFile = open("hmmoutput.txt","w",encoding="utf-8")

"""
sentence = "mister gray squirrels money"
wordsInSentence = sentence.split(' ')
wordsInSentence = [x for x in wordsInSentence if(x!='')]
"""

#Reading from test file
testFile = open(sys.argv[1],"r",encoding="utf-8")
content = testFile.readlines()
content = [x.rstrip("\n") for x in content]

for sentence in content:
    wordsInSentence = sentence.split(' ')
    wordsInSentence = [x for x in wordsInSentence if(x!='')]

    #Storing the probabilities calculated by running Viterbi algorithm
    viterbiProbabilities = []

    #Calculating probabilities for first word
    viterbiDict = {}
    for tag in tagDict:
        viterbiDict[tag] = {}
        if wordsInSentence[0] in emissionProbDict:
            if(float(emissionProbDict[wordsInSentence[0]][tag])!=float(0)):
                viterbiDict[tag]['q0'] = float(transitionProbDict['q0'][tag]) * float(emissionProbDict[wordsInSentence[0]][tag])
            else:
                viterbiDict[tag]['q0'] = float(0)
        else:
            viterbiDict[tag]['q0'] = float(transitionProbDict['q0'][tag])
        viterbiDict[tag]['q0'] = '{:.125f}'.format(viterbiDict[tag]['q0'])

    viterbiProbabilities.append(viterbiDict)

    #Calculating probabilities for subsequent words
    for i in range(1,len(wordsInSentence)):
        viterbiDict = {}
        for tag in tagDict:
            viterbiDict[tag] = {}
            for prevTag in tagDict:
                if wordsInSentence[i] in emissionProbDict:
                    if(not(float(emissionProbDict[wordsInSentence[i]][tag])==float(0) or float(max(viterbiProbabilities[i-1][prevTag].values()))==float(0))):
                        viterbiDict[tag][prevTag] = float(max(viterbiProbabilities[i-1][prevTag].values()))*float(transitionProbDict[prevTag][tag])*float(emissionProbDict[wordsInSentence[i]][tag])
                    else:
                        viterbiDict[tag][prevTag] = float(0)
                else:
                    viterbiDict[tag][prevTag] = float(max(viterbiProbabilities[i-1][prevTag].values()))*float(transitionProbDict[prevTag][tag])
                viterbiDict[tag][prevTag] = '{:.125f}'.format(viterbiDict[tag][prevTag])
        viterbiProbabilities.append(viterbiDict)


    #Calculating ending transitions
    viterbiDict = {}
    viterbiDict['r0'] = {}
    for tag in tagDict:
        if(float(max(viterbiProbabilities[len(wordsInSentence)-1][tag].items(),key=operator.itemgetter(1))[1])!=float(0)):
            viterbiDict['r0'][tag] = float(max(viterbiProbabilities[len(wordsInSentence)-1][tag].items(),key=operator.itemgetter(1))[1])*float(transitionProbDict[tag]['r0'])
        else:
            viterbiDict['r0'][tag] = float(0)
        viterbiDict['r0'][tag] = '{:.125f}'.format(viterbiDict['r0'][tag])

    viterbiProbabilities.append(viterbiDict)

    #Creating the backpointer list
    backPointer = []

    #Getting the last tag
    lastTagDict = viterbiProbabilities[len(viterbiProbabilities)-1]['r0']
    backPointer.insert(0,max(lastTagDict.items(),key=operator.itemgetter(1))[0])

    #Getting the previous tags
    for i in range(len(viterbiProbabilities)-2,0,-1):
        maxValue = max(viterbiProbabilities[i][backPointer[0]].items(),key=operator.itemgetter(1))[1]
        maxTags = [k for k,v in viterbiProbabilities[i][backPointer[0]].items() if v == maxValue]
        backPointer.insert(0,random.choice(maxTags))

    #Adding the tags to the words
    for i in range(len(wordsInSentence)):
        wordsInSentence[i] += "/"+backPointer[i]

    #Creating the tagged sentence
    sentence = ' '.join(wordsInSentence)

    #Writing to the output file
    outputFile.write(sentence+"\n")
