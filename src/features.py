import polars as pl
import nltk



class Features:
    def __init__(self, tokenizer, stop_words, speller, ner_model):
        self.tokenizer = tokenizer
        print('loaded model chkpt')
        self.STOP_WORDS = stop_words
        print('loaded stop words')
        self.speller = speller
        self.spacy_ner_model = ner_model
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
        # print('computing bigrams')
        return [" ".join(n) for n in nltk.ngrams(value, 2)]

    def compute_trigrams(self, value):
        # print('computing trigrams')
        return [" ".join(n) for n in nltk.ngrams(value, 3)]

    def get_ner_entities(self, value):
        # print('computing ner entities')
        model = self.spacy_ner_model
        ner_tokens = model(value)
        ner_ents = list(set([(" ".join([token.text.lower(), token.label_])) for token in ner_tokens.ents]))
        return ner_ents


def prompt_features(df: pl.LazyFrame, features: Features) -> pl.LazyFrame:
    df = df.with_columns(
				prompt_length = pl.col('prompt_text').map_elements(features.get_len, return_dtype=pl.Int64),
				prompt_tokens = pl.col('prompt_text').map_elements(features.get_tokens, return_dtype=pl.List(str))
    )
    return df

def summary_features(df: pl.LazyFrame, features: Features) -> pl.LazyFrame:
    df = df.with_columns(
				summary_length = pl.col('text').map_elements(features.get_len, return_dtype=pl.Int64),
				summary_tokens = pl.col('text').map_elements(features.get_tokens, return_dtype=pl.List(str)),
				misspell_counts = pl.col('text').map_elements(features.get_misspell_counts, return_dtype=pl.Int64))
    return df

def prompt_summary_features(df: pl.LazyFrame, features: Features) -> pl.LazyFrame :
    df = df.with_columns(
				lenght_ratio = pl.col('summary_length') / pl.col('prompt_length'),
				word_overlap_count = ((pl.col("prompt_tokens").list.set_intersection(features.STOP_WORDS))
						                 .list.set_intersection(
						                   (pl.col("summary_tokens").list.set_intersection(features.STOP_WORDS))
						                 ).list.len()),
				bigram_overlap_count = (pl.col('prompt_tokens').map_elements(features.compute_bigrams, return_dtype=pl.List(str))
						                .list.set_intersection(
						                    pl.col('summary_tokens').map_elements(features.compute_bigrams, return_dtype=pl.List(str))
						                ).list.len()),
				trigram_overlap_count = (pl.col('prompt_tokens').map_elements(features.compute_trigrams, return_dtype=pl.List(str))
						                    .list.set_intersection(
						                    pl.col('summary_tokens').map_elements(features.compute_trigrams, return_dtype=pl.List(str))
						                ).list.len()))
				# ner_overlap_count =  ( pl.col('prompt_text').map_elements(preproc.get_ner_entities, return_dtype=pl.List(str))
						    #               .list.set_intersection(
						    #               pl.col('text').map_elements(preproc.get_ner_entities, return_dtype=pl.List(str))
						    #           ).list.len()))
    return df