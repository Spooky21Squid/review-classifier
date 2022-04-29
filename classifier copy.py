from dotenv import load_dotenv
from pathlib import Path
import os
import math
import re

class Model:
    """A naive-bayes model of IMDB reviews"""

    def __init__(self, dataDirs, stopWordsPath, alpha=1):
        """Constructs a multinomial naive-bayes model using a Bag of Words approach. dataDirs - a list of directories 
        containing the sample reviews. stopWordsPath - A path to a file containing a list of stop words to remove from
        samples. alpha - The value for smoothing, default is 1"""

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

        for directory in dataDirs:
            for child in Path(directory).iterdir():
                if child.is_file():
                    #print(f"'{child.name}':'{child.read_text()}'\n")

                    label = int(re.search(r"_[\d]+" , child.name).group(0)[1:])  # Get label from the filename
                    text = child.read_text()
                    tokens = tokenise(text, self.stopWords)
                    # We have a label and some tokens, now add it to the model
                    self.__addToModel(label, tokens)
        
        self.x = len(self.vocab)  # The number of unique words in the vocabulary
        
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

            for word in histogram.keys():
                freq = histogram[word]
                adjustedFreq = freq + self.alpha
                histogram[word] = adjustedFreq / adjustedTotalWordsInClass
            
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

    
    def __addToModel(self, label, tokens):
        """Adds a single Sample to the model. A sample consists of a label, and a bag of words feature vector"""

        # How to add:
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
        
        probabilities = [self.getProbability(tokens, i) for i in self.totalClasses.keys()]

        maxP = probabilities[0]
        maxI = 0
        for i in range(1, len(probabilities)):
            if probabilities[i] > maxP:
                maxI = i
                maxP = probabilities[i]
        
        return maxI + 1  # Return the number of class, not the index (which is index + 1)

    
    def getProbability(self, tokens, i):
        """Calculates the probability of the sample being in the class i"""
        
        pClass = math.log(self.totalClasses[i] / self.total)  # The prior probability of the sample being in class i
        totalP = 0

        # Multiply totalP by the probability of each word in the sample being in class i
        for word, freq in tokens.items():
            for j in range(freq):
                totalP += math.log(self.wordTotals[i].get(word, self.zeroFrequencyProbabilities[i]))
            #totalP *= math.pow(self.wordTotals[i].get(word, self.zeroFrequencyProbabilities[i]), freq)

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
    dataDirs = [os.environ['TRAINPOSDIR'], os.environ['TRAINNEGDIR']]
    model = Model(dataDirs, os.environ['STOPWORDS'])
    print("Number of unique words in all samples: %d" % (len(model.vocab)))

    # Test Accuracy

    numCorrect = 0
    total = 0
    withinX = dict()  # Keys: difference, values: frequency

    stopWords = set()
    with open(os.environ['STOPWORDS']) as f:
        contents = f.readlines()
    for line in contents:
        stopWords.add(line.strip())

    for directory in dataDirs:
        for child in Path(directory).iterdir():
            if child.is_file():
                label = int(re.search(r"_[\d]+" , child.name).group(0)[1:])  # Get label from the filename
                text = child.read_text()
                tokens = tokenise(text, stopWords)

                # Predict
                predictedClass = model.predict(tokens)
                if predictedClass == label:
                    numCorrect += 1
                total += 1

                # Find difference
                diff = abs(predictedClass - label)
                if diff in withinX:
                    withinX[diff] += 1
                else:
                    withinX[diff] = 1
    
    accuracy = (numCorrect / total) * 100

    print("Accuracy: %.2f\n" % (accuracy))
    print("Differences:")
    for k,v in withinX.items():
        print("%d : %.2f" % (k, (v / total) * 100))