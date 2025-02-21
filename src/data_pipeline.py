import polars as pl
import yaml
import time

from models import init
from features import (
    prompt_features,
    summary_features,
    prompt_summary_features
)

# Initialize models
# load data
# clean data
# join data
# transform data
# feature eng
# write data


def data_pipeline():

    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    #Initialize models and features
    features = init(config)


    # load data
    prompts_path = config['data_stores']['root'] + config['data_stores']['prompts_train']
    summaries_path = config['data_stores']['root'] + config['data_stores']['summaries_train']
    train_prompts_df = pl.scan_csv(prompts_path)
    train_summaries_df = pl.scan_csv(summaries_path)

    # features
    train_prompts_features = train_prompts_df.pipe(prompt_features, features=features)
    train_summary_features = train_summaries_df.pipe(summary_features, features=features)
    train_df = train_summary_features.join(train_prompts_features, on='prompt_id', how='left')
    output = train_df.pipe(prompt_summary_features, features=features)
    t1 = time.time()

    return output.collect()

if __name__ == '__main__':
    t0 = time.time()
    output = data_pipeline()
    t1 = time.time()
    print(f"pipeline execution time took: {t1 - t0} secs")
    print(output.head(3))
