# The vocab consists of 89527 unique words

from os.path import exists
from dotenv import load_dotenv
import os
import math

class Model:
    """A naive-bayes model of IMDB reviews"""

    def __init__(self, dataPath, vocabPath, alpha=1):
        """Constructs a multinomial naive-bayes model using a Bag of Words approach. dataPath - the path to the bag
        of words in LIBSVM format. vocabPath - the path to the text file containing a new-line-delineated list of words.
        alpha - The value for smoothing, default is 1"""

        self.alpha = alpha  # The value to add to apply smoothing
        self.x = countLinesInFile(vocabPath)  # The number of unique words in the vocabulary
        self.totalClasses = {k:0 for k in range(1,11)}  # Total number of samples in each class (1-10)
        self.wordTotals = {k:dict() for k in range(1,11)}  # A histogram of the frequency of words in each class (When calculateProbabilities is called, frequencies are replaced by probabilities)
        #self.wordProbabilities = {k:dict() for k in range(1,11)}  # A dictionary of the probabilities of finding words in each class
        self.zeroFrequencyProbabilities = {k:0 for k in range(1,11)}  # The probability of seeing a word in a class that didn't occur in the training set, adjusted for smoothing, for each class
        self.vocab = set()  # A set of all word IDs found in the test samples

        # Could just have a method for getting this value - add all the self.totalClasses values?
        self.total = 0  # Total number of samples

        with open(dataPath) as f:
            content = f.readlines()
        
        for line in content:
            s = Sample(line)
            
            self.__addToModel(s)
        
        self.__calculateProbabilities()
        
    
    def __calculateProbabilities(self):
        """Calculates the probabilities of seeing words given a class of sample. alpha - the 
        value to add to probabilities to apply smoothing, default is 1 (laplace smoothing)"""

        for class_, histogram in self.wordTotals.items():

            print("Class %d:" % (class_))
            totalWordsInClass = 0
            for freq in histogram.values():
                totalWordsInClass += freq
            print("total words: %d" % (totalWordsInClass))
            
            # Adjust the total word count for alpha
            adjustedTotalWordsInClass = totalWordsInClass + self.alpha * self.x
            print("adjusted total words after smoothing: %d" % (adjustedTotalWordsInClass))

            for wordID in histogram.keys():
                freq = histogram[wordID]
                adjustedFreq = freq + self.alpha
                histogram[wordID] = adjustedFreq / adjustedTotalWordsInClass
            
            # Calculate the probability of a word that doesnt appear in the class

            numOfZeroFreqs = self.x - len(histogram)  # Number of unique words that aren't in this class
            probZeroFreq = self.alpha / adjustedTotalWordsInClass  # P of seeing a single non-frequency word
            self.zeroFrequencyProbabilities[class_] = probZeroFreq
            sumOfZeroFreqs = probZeroFreq * numOfZeroFreqs  # Sum of all those probabilities (Should be 1 - the sum of P of all words in the class)
            print("Number of words that don't appear: %d" % (numOfZeroFreqs))
            print("Probability of seeing a zero-frequency word: %.15f" % (probZeroFreq))
            
            sumOfProbabilities = 0
            for p in histogram.values():
                sumOfProbabilities += p
            sumOfProbabilities += sumOfZeroFreqs
            print("sum of probabilities: %.5f" % (sumOfProbabilities), end="\n\n")  # Should be 1

    
    def __addToModel(self, sample):
        """Adds a single Sample to the model. A sample consists of a label, and a bag of words feature vector"""

        # How to add:
        # self.wordTotals (
        #   Key: class
        #   Value: dict (
        #       Key: wordID
        #       Value: frequency

        for wordID, freq in sample.features.items():
            self.vocab.add(wordID)  # Add the word ID to the vocab
            if wordID not in self.wordTotals[sample.label]:
                self.wordTotals[sample.label][wordID] = freq
            else:
                self.wordTotals[sample.label][wordID] += freq
        
        self.total += 1
        self.totalClasses[sample.label] += 1
    

    def predict(self, sample):
        """Predicts the class (1-10) of a review using naive-bayes"""
        
        probabilities = [self.getProbability(sample, i) for i in range(1, 11)]

        maxP = probabilities[0]
        maxI = 0
        for i in range(1, len(probabilities)):
            if probabilities[i] > maxP:
                maxI = i
                maxP = probabilities[i]
        
        return maxI + 1  # Return the number of class, not the index (which is index + 1)

    
    def getProbability(self, sample, i):
        """Calculates the probability of the sample being in the class i"""
        
        pClass = self.totalClasses[i] / self.total  # The prior probability of the sample being in class i
        totalP = pClass

        # Multiply totalP by the probability of each word in the sample being in class i
        for wordID, freq in sample.features.items():
            #totalP *= math.pow(self.wordTotals[i][wordID] self.wordTotals[i].get(wordID, self.zeroFrequencyProbabilities[i]), freq)
            totalP *= math.pow(self.wordTotals[i].get(wordID, self.zeroFrequencyProbabilities[i]), freq)

        return totalP


class Sample:
    """Represents a single review as a bag of words with a label"""

    def __init__(self, line):
        """Creates a sample from a line of text from the LIBSVM bag of words file, where label - an integer 
        score from 1 to 10. features - a dictionary of wordID (int, >= 0) to frequency (int, >= 1)."""

        label, values = line.split(' ', 1)
        self.label = int(label)
        self.features = dict()
        
        for e in values.split():
            wordID, freq = e.split(':')
            self.features[int(wordID)] = int(freq)
    
    
    def getLabel(self):
        return self.label
    
    def getFeatures(self):
        return self.features
    
    def getFreq(self, wordID):
        if not wordID in self.features:
            return 0
        else:
            return self.features[wordID]
    
    def __str__(self):
        s = "%d" % (self.label)
        for wordID, freq in self.features.items():
            s += " %d:%d" % (wordID, freq)
        s += '\n'
        return s


def countLinesInFile(path):
    """Reads a file and returns the number of non-blank lines"""
    x = 0

    with open(path) as f:
        c = f.readlines()
        for line in c:
            if len(line.strip()) != 0:
                x += 1
    
    return x


def readBOW(path):
    """Reads the bag of words file in LIBSVM format (<label> <word ID>:<frequency>) and puts it in a list"""
    
    samples = list()

    with open(path) as f:
        content = f.readlines()
    
    for line in content:
        samples.append(Sample(line))
    
    return samples


if __name__ == "__main__":

    load_dotenv()  # take environment variables from .env.
    
    # Create the model
    model = Model(os.environ['TRAINDATA'], os.environ['VOCABFILE'])
    print("Number of unique words in all samples: %d" % (len(model.vocab)))

    # Test Accuracy

    # Get the test data
    samples = readBOW(os.environ['TESTDATA'])
    numCorrect = 0
    total = 0

    for sample in samples:
        predictedClass = model.predict(sample)
        actualClass = sample.getLabel()
        if predictedClass == actualClass:
            numCorrect += 1
        total += 1
    
    accuracy = (numCorrect / total) * 100

    print("Accuracy: %.2f" % (accuracy))