from logging import NullHandler
import pandas as pd
import numpy as np
import random
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from pyparsing import null_debug_action
from trectools import TrecQrel
import trec_car.read_data

class Preprocessor(object):
    def __init__(self):
        self.train