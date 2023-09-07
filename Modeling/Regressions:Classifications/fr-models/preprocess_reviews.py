import sys
import pandas as pd
import spacy
import srsly
from joblib import Parallel, delayed
from tqdm import tqdm

base_dir = '../'
sys.path.insert(1, base_dir)
import src.config as cfg
import src.utils as utils
from src.cache_data import load_cache

nlp = spacy.load("en_core_web_sm")

# Add contractions missing in spaCy 3.x
# https://github.com/explosion/spaCy/issues/7014

contraction_patterns = srsly.read_json(f"{base_dir}data/ref/ar_patterns.json")

nlp.remove_pipe("attribute_ruler")
ar = nlp.add_pipe("attribute_ruler", before="lemmatizer")
ar.add_patterns(contraction_patterns)

custom_proper_nouns = ['lazard', 'lam', 'fa', 'hr']
keep_punctuation_symbols = [".", ",", "'"]


def load_raw_reviews(all_raw_files):
    """
    Read in each batch of reviews in all_raw_files (sorted from newest to oldest), keeping only repeated Review ID's
    from the newest batch
    :param all_raw_files: lst
    :return: pd.DataFrame
    """
    # all_raw_files is sorted from newest to oldest
    df_reviews_raw = pd.DataFrame()
    for raw_file in all_raw_files:
        raw_filepath = f'{base_dir}data/raw/{raw_file}'
        if '.csv' in raw_filepath:
            df_batch = pd.read_csv(raw_filepath)
        elif '.h5' in raw_filepath:
            df_batch = pd.read_hdf(raw_filepath)
        else:
            raise Exception(f"Expected file ending of '.csv' or '.h5' - got {raw_filepath} instead")

        if not df_reviews_raw.empty:
            df_not_already_included = df_batch.loc[~df_batch['Review ID'].isin(df_reviews_raw['Review ID'].unique())]
        else:
            df_not_already_included = df_batch.copy()
        df_reviews_raw = pd.concat([df_reviews_raw, df_not_already_included]).reset_index(drop=True)
        utils.logger.info(f"Adding {df_not_already_included.shape[0]}/{df_batch.shape[0]} new reviews from {raw_file}")

    # Ensure all LDAG reviews are removed
    df_reviews_raw = df_reviews_raw.loc[df_reviews_raw['Reviewee Department'] != 'Data Analytics Group'].copy()

    return df_reviews_raw


def preprocess_text(doc):
    """
    Remove stop words, remove proper nouns (both spaCy-defined and custom), and remove punctuation if not in custom list
    :param doc: spacy.tokens.doc.Doc
    :return:str
    """
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop
                         and (token.pos_ != 'PROPN' and token.text not in custom_proper_nouns)
                         and (token.pos_ != 'PUNCT' or token.text in keep_punctuation_symbols)]
    return " ".join(lemmatized_tokens)


def preprocess_all_reviews(df_reviews):
    """
    Preprocess all reviews (both text + non-text)
    :param df_reviews: pd.DataFrame
    :return: pd.DataFrame
    """
    # Create Hash Values for Unique Questions
    df_reviews['Question ID'] = (df_reviews['Question Text']
                                 .apply(hash) % (10 ** 4)
                                 ).astype(str)

    # trim whitespace
    df_reviews['Reviewee Business Unit'] = df_reviews['Reviewee Business Unit'].str.strip()

    # Fill in missing years to 2021
    df_reviews.loc[df_reviews['Review Cycle Period'].isna(), 'Review Cycle Period'] = 2021

    return df_reviews


def preprocess_text_reviews(df_reviews):
    """
    Pre-processes review texts
    :param df_reviews: pd.DataFrame
    :return: pd.DataFrame
    """
    if df_reviews.empty:
        utils.logger.warning(f"Got no text reviews, returning empty dataframe")
        return pd.DataFrame()
    else:
        utils.logger.info(f"Preprocessing {df_reviews.shape[0]} text reviews")

    df_reviews = df_reviews.dropna(subset=['Response Text']).copy()
    df_reviews['Processed Response Text'] = (df_reviews['Response Text'].astype(str)
                                             .str.replace(r"\*EMPLOYEE FIRST NAME\*", "", regex=True)
                                             .str.replace(r"[\s]+", " ", regex=True)
                                             .str.lower().str.strip()
                                             )

    texts = df_reviews['Processed Response Text'].values
    df_reviews['Processed Response Text'] = Parallel(n_jobs=-1, prefer='threads')(
        delayed(preprocess_text)(doc) for doc in tqdm(nlp.pipe(texts, disable=['parser']), total=len(texts)))
    # Vectorized operations to speed up
    df_reviews['Processed Response Text'] = (df_reviews['Processed Response Text'].astype(str)
                                             .str.replace(rf"[^a-zA-Z{''.join(keep_punctuation_symbols)}]+", " ",
                                                          regex=True))
    # Substitute all " ." with "." for example
    for punct in keep_punctuation_symbols:
        df_reviews['Processed Response Text'] = (df_reviews['Processed Response Text']
                                                 .str.replace(rf"\s\{punct}", punct, regex=True)
                                                 .str.replace(rf"^\{punct}", "", regex=True)
                                                 .str.strip())

    return df_reviews

@load_cache
def preprocess_pipeline(df_reviews_raw_all):
    df_reviews_processed_all = preprocess_all_reviews(df_reviews_raw_all)
    df_text, df_non_text = utils.split_reviews(df_reviews_processed_all)

    df_text = preprocess_text_reviews(df_text)
    df_reviews_processed = pd.concat([df_text, df_non_text], axis=0)

    df_reviews_processed.to_csv(f'{base_dir}data/derived/Preprocessed HR {cfg.VERSION_NUMBER}.csv',
                                index=False)

    return df_reviews_processed


if __name__ == "__main__":
    df_raw = load_raw_reviews(cfg.RAW_HR_FILE)
    preprocess_pipeline(df_raw)
