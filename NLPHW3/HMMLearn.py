#Third attempt
import sys

inputFile = open(sys.argv[1],"r",encoding="utf-8")
content = inputFile.readlines()
content = [x.rstrip("\n") for x in content]

"""
content = ["mister/NN jones/NN","money/NN talks/VB","mister/NN runs/VB","drink/VB tea/NN",
"blue/JJ gray/JJ","mister/NN smith/NN","gray/NN wins/VB","princess/NN royal/JJ","cook/VB potatoes/NN",
"happy/JJ squirrels/NN","color/NN gray/NN","squirrels/NN drink/VB","squirrels/VB food/NN",
"fix/VB things/NN","yellow/JJ red/JJ","squirrels/NN eat/VB","mister/NN healthy/JJ","earn/VB money/NN",
"gray/JJ squirrels/NN","mellow/JJ green/JJ"]
"""

#Creating dictionaries for counting number of tags, words, transitions
tagDictionary = {}
wordDictionary = {}
transitionDictionary = {}
transitionDictionary['q0'] = {}

#Counting for every sentence
for sentence in content:
    wordsInSentence = sentence.split(' ')
    wordsInSentence = [x for x in wordsInSentence if(x!='')]
    #Taking every word in the sentence
    for i in range(len(wordsInSentence)):
        word = wordsInSentence[i].rsplit('/',1)[0]
        tag = wordsInSentence[i].rsplit('/',1)[1]

        #Adding count word and tag to the words dictionary
        if word in wordDictionary:
            if tag in wordDictionary[word]:
                wordDictionary[word][tag] += 1
            else:
                wordDictionary[word][tag] = 1
        else:
            wordDictionary[word] = {}
            wordDictionary[word][tag] = 1

        #Adding count of tags in tag dictionary and transition tag dictionary
        if tag in tagDictionary:
            tagDictionary[tag] += 1
        else:
            tagDictionary[tag] = 1

        #Counting the transitions
        if(len(wordsInSentence)==1):
            if tag in transitionDictionary['q0']:
                transitionDictionary['q0'][tag] += 1
            else:
                transitionDictionary['q0'][tag] = 1

            if tag in transitionDictionary:
                if 'r0' in transitionDictionary[tag]:
                    transitionDictionary[tag]['r0'] += 1
                else:
                    transitionDictionary[tag]['r0'] = 1
            else:
                transitionDictionary[tag] = {}
                transitionDictionary[tag]['r0'] = 1
        else:
            if(i==len(wordsInSentence)-1):
                if tag in transitionDictionary:
                    if 'r0' in transitionDictionary[tag]:
                        transitionDictionary[tag]['r0'] += 1
                    else:
                        transitionDictionary[tag]['r0'] = 1
                else:
                    transitionDictionary[tag] = {}
                    transitionDictionary[tag]['r0'] = 1
            else:
                nextTag = wordsInSentence[i+1].rsplit('/',1)[1]
                if tag in transitionDictionary:
                    if nextTag in transitionDictionary[tag]:
                        transitionDictionary[tag][nextTag] += 1
                    else:
                        transitionDictionary[tag][nextTag] = 1
                else:
                    transitionDictionary[tag] = {}
                    transitionDictionary[tag][nextTag] = 1
                if(i==0):
                    if tag in transitionDictionary['q0']:
                        transitionDictionary['q0'][tag] += 1
                    else:
                        transitionDictionary['q0'][tag] = 1

#Adding zeroes to word dictionary and transition dictionary
for tag in tagDictionary:
    for word in wordDictionary:
        if(tag not in wordDictionary[word]):
            wordDictionary[word][tag] = 0

    if tag not in transitionDictionary:
        transitionDictionary[tag] = {}

for tag in tagDictionary:
    for transTag in transitionDictionary:
        if(tag not in transitionDictionary[transTag]):
            transitionDictionary[transTag][tag] = 0

    if tag not in transitionDictionary['q0']:
        transitionDictionary['q0'][tag] = 0

    if 'r0' not in transitionDictionary[tag]:
        transitionDictionary[tag]['r0'] = 0

#Calculating emission probabilities
emissionProbDictionary = {}
for word in wordDictionary:
    emissionProbDictionary[word] = {}
    for tag in tagDictionary:
        emissionProbDictionary[word][tag] = float(wordDictionary[word][tag])/float(tagDictionary[tag])
        emissionProbDictionary[word][tag] = '{:.10f}'.format(emissionProbDictionary[word][tag])

#Calculating transition probabilities and applying smoothing
transitionProbDictionary = {}
for tag in transitionDictionary:
    transitionProbDictionary[tag] = {}
    for transTag in transitionDictionary[tag]:
        transitionProbDictionary[tag][transTag] = float(transitionDictionary[tag][transTag]+1)/float(sum(transitionDictionary[tag].values())+len(tagDictionary))
        transitionProbDictionary[tag][transTag] = '{:.10f}'.format(transitionProbDictionary[tag][transTag])

#Writing the transition probabilities, emission probabilities and tags to hmmmodel.txt
modelFile = open("hmmmodel.txt","w",encoding="utf-8")
modelFile.write(str(transitionProbDictionary)+"\n")
modelFile.write(str(emissionProbDictionary)+"\n")
modelFile.write(str(tagDictionary)+"\n")
