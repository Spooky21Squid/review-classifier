from pathlib import Path
import os
from dotenv import load_dotenv
import re

load_dotenv()

def readReviews(directory):
    """Reads a directory of reviews and outputs a list of cleaned, tokenised bag of words samples"""
    

    for child in Path(directory).iterdir():
        if child.is_file():
            #print(f"'{child.name}':'{child.read_text()}'\n")

            # get label
            label = int(re.search(r"_[\d]+" , child.name).group(0)[1:])
            text = child.read_text()

readReviews(os.environ['TRAINPOSDIR'])