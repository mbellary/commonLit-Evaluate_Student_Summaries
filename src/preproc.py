# Credit:
# This work is based on the top  Kernel from the competition : https://www.kaggle.com/code/tsunotsuno/debertav3-lgbm-no-autocorrect

import polars as pl
import transformers
import nltk
import spacy
import argparse
import timeit
from transformers import AutoTokenizer
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import time


class Preprocessor:
    def __init__(self, model_chkpt):
        self.tokenizer = AutoTokenizer.from_pretrained(model_chkpt)
        print('loaded model chkpt')
        self.STOP_WORDS = list(set(stopwords.words("english")))
        print('loaded stop words')
        self.speller = SpellChecker()
        self.spacy_ner_model = spacy.load("en_core_web_sm")
        print('loaded ner model')
        
    def get_len(self, value):
        return len(self.tokenizer(value)['input_ids'])
    
    def get_tokens(self, value):
        return self.tokenizer.convert_ids_to_tokens(self.tokenizer(value)['input_ids'], skip_special_tokens=True)

    def get_misspell_counts(self, value):
        tokens = value.split()
        misspell_counts = len(set(self.speller.unknown(tokens)))
        return misspell_counts

    def compute_bigrams(self, value):
    	#print('computing bigrams')
    	return [" ".join(n) for n in nltk.ngrams(value, 2)]

    def compute_trigrams(self, value):
    	#print('computing trigrams')
    	return [" ".join(n) for n in nltk.ngrams(value, 3)]

    def get_ner_entities(self, value):
    	#print('computing ner entities')
    	model = self.spacy_ner_model
    	ner_tokens = model(value)
    	ner_ents = list(set([(" ".join([token.text.lower(), token.label_])) for token in ner_tokens.ents]))
    	return ner_ents



def main(dir_path, model_path):
	start_msg = "Reading data files.."
	end_msg = "completed processing data files"
	print(start_msg)
	train_prompts_df = pl.read_csv(dir_path + "datasets/commonlit-evaluate-student-summaries/prompts_train.csv")
	train_summaries_df = pl.read_csv(dir_path + "datasets/commonlit-evaluate-student-summaries/summaries_train.csv")
	preproc = Preprocessor(model_path)
	train_prompts_df = train_prompts_df.with_columns(
						    prompt_length = pl.col('prompt_text').map_elements(preproc.get_len, return_dtype=pl.Int64),
						    prompt_tokens = pl.col('prompt_text').map_elements(preproc.get_tokens, return_dtype=pl.List(str)))
	print('computed prompt tokens and len')
	train_summaries_df = train_summaries_df.with_columns(
						    summary_length = pl.col('text').map_elements(preproc.get_len, return_dtype=pl.Int64),
						    summary_tokens = pl.col('text').map_elements(preproc.get_tokens, return_dtype=pl.List(str)),
						    misspell_counts = pl.col('text').map_elements(preproc.get_misspell_counts, return_dtype=pl.Int64))
	print('computed summary tokens and len')
	train_df = train_summaries_df.join(train_prompts_df, on='prompt_id', how='left')
	print('joined summary and prompt')
	train_df = train_df.with_columns(
						    lenght_ratio = pl.col('summary_length') / pl.col('prompt_length'),
						    word_overlap_count = ((pl.col("prompt_tokens").list.set_intersection(preproc.STOP_WORDS))
						                             .list.set_intersection(                             
						                          (pl.col("summary_tokens").list.set_intersection(preproc.STOP_WORDS))
						                     ).list.len()),
						    bigram_overlap_count = (pl.col('prompt_tokens').map_elements(preproc.compute_bigrams, return_dtype=pl.List(str))
						                              .list.set_intersection(
						                            pl.col('summary_tokens').map_elements(preproc.compute_bigrams, return_dtype=pl.List(str))
						                    ).list.len()),
						    trigram_overlap_count = (pl.col('prompt_tokens').map_elements(preproc.compute_trigrams, return_dtype=pl.List(str))
						                          .list.set_intersection(
						                        pl.col('summary_tokens').map_elements(preproc.compute_trigrams, return_dtype=pl.List(str))
						                ).list.len()))
						    # ner_overlap_count =  ( pl.col('prompt_text').map_elements(preproc.get_ner_entities, return_dtype=pl.List(str))
						    #                          .list.set_intersection(
						    #                      pl.col('text').map_elements(preproc.get_ner_entities, return_dtype=pl.List(str))
						    #             ).list.len()))
	print('done!')

	print(end_msg)
	print(train_df.head(5))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess CommonLit-ESS training data')
	parser.add_argument('--dir_path', metavar='path', required=True, help='the path to data files')
	parser.add_argument('--model_path', metavar='path', required=True, help='the path to model checkpoint files')
	args = parser.parse_args()
	t0 = time.time()
	main(args.dir_path, args.model_path)
	t1 = time.time()
	print(f"pipeline execution time took: {t1 - t0} secs")
	#print(timeit.timeit("main(args.dir_path, args.model_path)"))