from dotenv import load_dotenv
from pathlib import Path
import os
import math
import re

class Model:
    """A naive-bayes model trained on the set of IMDB reviews"""

    def __init__(self, posDir, negDir, stopWordsPath, alpha=1):
        """Constructs a multinomial naive-bayes model using a Bag of Words approach with logarithmic probability and stopwords. 
        dataDirs - a list of directories containing the sample reviews. stopWordsPath - A path to a file containing a list of 
        stop words to remove from samples. alpha - The value for smoothing, default is 1"""

        # Read the list of stop words from a file
        self.stopWords = set()  # A set of stopwords
        with open(stopWordsPath) as f:
            contents = f.readlines()
        for line in contents:
            self.stopWords.add(line.strip())

        self.alpha = alpha  # The value to add to apply smoothing
        self.totalClasses = dict()  # Total number of samples in each class (className:frequency)
        self.wordTotals = dict()  # (className:dict()) A histogram of the frequency of words in each class (When calculateProbabilities is called, frequencies are replaced by probabilities)
        self.zeroFrequencyProbabilities = dict()  # The probability of seeing a word in a class that didn't occur in the training set, adjusted for smoothing, for each class
        self.vocab = set()  # A set of all words found in the test samples, minus the stop words
        self.total = 0  # Total number of samples

        # Incrementally add each training sample to the model by adding it to the frequency histogram
        for child in Path(posDir).iterdir():  # positive samples (7 - 10)
            if child.is_file():
                text = child.read_text()
                tokens = tokenise(text, self.stopWords)
                self.__addToModel('positive', tokens)
        
        for child in Path(negDir).iterdir():  # negative samples (0 - 4)
            if child.is_file():
                text = child.read_text()
                tokens = tokenise(text, self.stopWords)
                self.__addToModel('negative', tokens)
        
        self.x = len(self.vocab)  # The number of unique words in the vocabulary
        
        # Convert the histogram to a mapping from words to probabilities
        self.__calculateProbabilities()
        
    
    def __calculateProbabilities(self):
        """Calculates the probabilities of seeing words in each sample given the training data set"""

        for class_, histogram in self.wordTotals.items():

            #print("Class %d:" % (class_))
            totalWordsInClass = 0
            for freq in histogram.values():
                totalWordsInClass += freq
            #print("total words: %d" % (totalWordsInClass))
            
            # Adjust the total word count for alpha
            adjustedTotalWordsInClass = totalWordsInClass + self.alpha * self.x
            #print("adjusted total words after smoothing: %d" % (adjustedTotalWordsInClass))

            for word in histogram.keys():
                freq = histogram[word]
                adjustedFreq = freq + self.alpha
                histogram[word] = adjustedFreq / adjustedTotalWordsInClass
            
            # Calculate the probability of a word that doesnt appear in the class

            numOfZeroFreqs = self.x - len(histogram)  # Number of unique words that aren't in this class
            probZeroFreq = self.alpha / adjustedTotalWordsInClass  # P of seeing a single non-frequency word
            self.zeroFrequencyProbabilities[class_] = probZeroFreq
            sumOfZeroFreqs = probZeroFreq * numOfZeroFreqs  # Sum of all those probabilities (Should be 1 - the sum of P of all words in the class)
            #print("Number of words that don't appear: %d" % (numOfZeroFreqs))
            #print("Probability of seeing a zero-frequency word: %.15f" % (probZeroFreq))
            
            sumOfProbabilities = 0
            for p in histogram.values():
                sumOfProbabilities += p
            sumOfProbabilities += sumOfZeroFreqs
            #print("sum of probabilities: %.5f" % (sumOfProbabilities), end="\n\n")  # Should be 1

    
    def __addToModel(self, label, tokens):
        """Adds a single sample to the model. A sample consists of a label, and tokens - a bag of words feature vector"""

        # self.wordTotals (
        #   Key: class
        #   Value: dict (
        #       Key: wordID
        #       Value: frequency

        for word, freq in tokens.items():
            self.vocab.add(word)  # Add the word to the vocab

            if label not in self.wordTotals:
                self.wordTotals[label] = dict()

            if word not in self.wordTotals[label]:
                self.wordTotals[label][word] = freq
            else:
                self.wordTotals[label][word] += freq
        
        self.total += 1

        if label not in self.totalClasses:
            self.totalClasses[label] = 1
        else:
            self.totalClasses[label] += 1
        
        if label not in self.zeroFrequencyProbabilities:
            self.zeroFrequencyProbabilities[label] = 0
    

    def predict(self, tokens):
        """Predicts the class of a review using naive-bayes"""
        
        # Choose the class with the maximum probability

        classes = [i for i in self.totalClasses.keys()]
        classes.sort()

        probabilities = [self.getProbability(tokens, i) for i in classes]
        #probabilities = [self.getProbability(tokens, i) for i in self.totalClasses.keys()]

        maxP = probabilities[0]
        maxI = 0
        for i in range(1, len(probabilities)):
            if probabilities[i] > maxP:
                maxI = i
                maxP = probabilities[i]
        
        #return maxI + 1  # Return the number of class, not the index (which is index + 1)
        return classes[maxI]  # Return the class with the highest probability

    
    def getProbability(self, tokens, i):
        """Calculates the probability of a sample being in the class i"""
        
        pClass = math.log(self.totalClasses[i] / self.total)  # The prior probability of the sample being in class i
        totalP = 0

        # Multiply totalP by the probability of each word in the sample being in class i
        for word, freq in tokens.items():
            for j in range(freq):
                #totalP += math.log(self.wordTotals[i].get(word, self.zeroFrequencyProbabilities[i]))
                totalP += math.log(self.wordTotals[i].get(word, 1))  # Don't include words that don't show up in the class

        totalP *= pClass
        return totalP


def tokenise(text, stopWords):
    """Cleans and tokenises a piece of text"""

    tokens = dict()
    text = re.sub(r"<.*>",'', text.lower())  # Remove html tags

    # Split into words and remove apostrophe from plurals
    words = [x.replace("'s", "s") for x in re.findall(r"[\w'-]+", text)]

    # Remove stop words
    cleanedWords = [x for x in words if x not in stopWords]

    for word in cleanedWords:
        if word in tokens:
            tokens[word] += 1
        else:
            tokens[word] = 1

    return tokens


if __name__ == "__main__":

    load_dotenv()  # take environment variables from .env.
    
    # Create the model
    model = Model(os.environ['TRAINPOSDIR'], os.environ['TRAINNEGDIR'], os.environ['STOPWORDS'], 1)
    print("Number of unique words in all samples: %d" % (len(model.vocab)))

    # Test Accuracy
    numberOfTestClasses = dict()

    numCorrect = 0
    numPredictedPos = 0
    numPredictedNeg = 0
    numFalsePositives = 0
    numFalseNegatives = 0
    total = 0

    stopWords = set()
    with open(os.environ['STOPWORDS']) as f:
        contents = f.readlines()
    for line in contents:
        stopWords.add(line.strip())

    testDirs = [(os.environ['TESTPOSDIR'], 'positive'), (os.environ['TESTNEGDIR'], 'negative')]
    for directory in testDirs:
        for child in Path(directory[0]).iterdir():
            if child.is_file():
                label = directory[1]
                text = child.read_text()
                tokens = tokenise(text, stopWords)

                # Count the test sample
                if label in numberOfTestClasses:
                    numberOfTestClasses[label] += 1
                else:
                    numberOfTestClasses[label] = 1

                # Predict
                predictedClass = model.predict(tokens)
                total += 1
                if predictedClass == label:
                    numCorrect += 1
                else:
                    if predictedClass == 'positive':
                        numFalsePositives += 1
                    else:
                        numFalseNegatives += 1
                if predictedClass == 'positive':
                    numPredictedPos += 1
                else:
                    numPredictedNeg += 1
    
    accuracy = (numCorrect / total) * 100

    print("Accuracy: %.2f\n" % (accuracy))
    
    print("Number of samples in different classes:")
    for k,v  in numberOfTestClasses.items():
        print("%s : %d" % (k, v))
    
    print("Number of predicted positive samples: %d (%.2f)" % (numPredictedPos, ((numPredictedPos / total) * 100)))
    print("Number of correct positives: %d (%.2f)" % (numPredictedPos - numFalsePositives, (((numPredictedPos - numFalsePositives) / total) * 100)))
    print("Number of false positives: %d (%.2f)" % (numFalsePositives, ((numFalsePositives / total) * 100)))
    print("Number of predicted negative samples: %d (%.2f)" % (numPredictedNeg, ((numPredictedNeg / total) * 100)))
    print("Number of correct negatives: %d (%.2f)" % (numPredictedNeg - numFalseNegatives, (((numPredictedNeg - numFalseNegatives) / total) * 100)))
    print("Number of false negatives: %d (%.2f)" % (numFalseNegatives, ((numFalseNegatives / total) * 100)))


