import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.cache_data import load_cache
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

base_dir = '../'
sys.path.insert(1, base_dir)

import src.config as cfg
import src.utils as utils


def get_sentiment_scores(df_reviews, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Call sentiment model_name on df_reviews['Response Text']
    :param df_reviews: pd.DataFrame
    :param model_name: str
    :return: pd.DataFrame
    """
    if df_reviews.empty:
        utils.logger.warning("Got no text reviews, returning empty dataframe")
        return pd.DataFrame()
    else:
        utils.logger.info(f"Getting sentiment scores for {df_reviews.shape[0]} reviews")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def get_label_probs(raw_texts):
        if type(raw_texts) != list:
            raw_texts = [raw_texts]

        tokenized_texts = tokenizer(raw_texts, truncation=True, padding=True, return_tensors="pt")
        label_logits = model(**tokenized_texts)
        label_probs = torch.softmax(label_logits[0], dim=1).tolist()
        return label_probs

    # Mask "*EMPLOYEE FIRST NAME*" for tokenizer
    df_reviews['Response Text'] = df_reviews['Response Text'].str.replace("\*EMPLOYEE FIRST NAME\*",
                                                                          tokenizer.mask_token).str.strip()

    all_label_probs = Parallel(n_jobs=-1, prefer="threads")(delayed(get_label_probs)(row['Response Text'])
                                                            for rix, row in
                                                            tqdm(df_reviews.iterrows(), total=df_reviews.shape[0]))

    all_label_probs = np.concatenate(all_label_probs)

    for class_number in range(all_label_probs.shape[1]):
        df_reviews[f'BERT Class {class_number + 1} Probability'] = all_label_probs[:, class_number]

    # Dot Product of probabilities/classes for expected value
    df_reviews['BERT Sentiment Score'] = sum(
        [df_reviews[f'BERT Class {i} Probability'] * i for i in np.arange(1, 5 + 1, 1)])

    df_reviews['Low Interaction'] = (((df_reviews['Response Text'].str.split(" ").apply(len) < 30) &
                                      (df_reviews['Response Text'].str.lower().str.contains(
                                          "n\/a|n\.a|low interact[^\s]+|no comment|none|not applicable", regex=True))) |
                                     (df_reviews['Response Text'].str.split(" |\n").apply(len) <= 2)).astype(int)

    return df_reviews

@load_cache
def sentiment_analysis_pipeline(df):
    df_text, df_non_text = utils.split_reviews(df)
    df_text = get_sentiment_scores(df_text.copy(), model_name="nlptown/bert-base-multilingual-uncased-sentiment")
    df_reviews_sentiment = pd.concat([df_text, df_non_text], axis=0)
    df_reviews_sentiment.to_csv(f'{base_dir}data/derived/Sentiment Analysis HR {cfg.VERSION_NUMBER}.csv',
                                index=False)
    return df_reviews_sentiment


if __name__ == "__main__":
    df_reviews_raw = pd.read_csv(f'{base_dir}data/derived/Preprocessed HR incl Topic Counts {cfg.VERSION_NUMBER}.csv')
    sentiment_analysis_pipeline(df_reviews_raw)
