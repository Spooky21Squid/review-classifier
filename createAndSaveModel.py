import classifier
import pickle
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()  # take environment variables from .env.
    
# Create the model
print("Creating the model")
model = classifier.Model(os.environ['TRAINPOSDIR'], os.environ['TRAINNEGDIR'], os.environ['STOPWORDS'], 1)
print("Model created")

# Save the model
try:
    with open("model.pickle", "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print("Couldnt save model:", e)
    quit()
print("Saved the model")

# Load and test model
try:
    with open("model.pickle", "rb") as file:
        m = pickle.load(file)
except Exception as e:
    print("Couldn't load model:", e)
    quit()
print("Loaded the model")

stopWords = set()
with open(os.environ['STOPWORDS']) as f:
    contents = f.readlines()
for line in contents:
    stopWords.add(line.strip())

total = 0
numCorrect = 0
testDirs = [(os.environ['TESTPOSDIR'], 'positive'), (os.environ['TESTNEGDIR'], 'negative')]
for directory in testDirs:
    for child in Path(directory[0]).iterdir():
        if child.is_file():
            label = directory[1]
            text = child.read_text()
            tokens = classifier.tokenise(text, stopWords)

            # Predict
            predictedClass = model.predict(tokens)
            total += 1
            if predictedClass == label:
                numCorrect += 1
accuracy = (numCorrect / total) * 100

print("Accuracy: %.2f\n" % (accuracy))