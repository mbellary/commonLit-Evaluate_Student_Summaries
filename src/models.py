
import nltk
import spacy
import argparse
import timeit

from pyexpat import features
from transformers import AutoTokenizer
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from features import Features

def init(config):
    tokenizer = AutoTokenizer.from_pretrained(config['hf_models']['path'])
    STOP_WORDS = list(set(stopwords.words("english")))
    speller = SpellChecker()
    spacy_ner_model = spacy.load("en_core_web_sm")
    features = Features(tokenizer, STOP_WORDS, speller, spacy_ner_model)

    return features