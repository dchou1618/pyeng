import sys
from scipy.stats import zscore
import pandas as pd
import numpy as np

sys.path.insert(1, '../')
from src.utils import min_max_scale, split_reviews, logger
from src.config import BASE_DIR, VERSION_NUMBER


def _debias(df, by, col, strategy='mean'):
    if strategy == 'mean':
        scores = df.groupby(by)[col].transform(lambda x: x - x.mean())
    elif strategy == 'median':
        scores = df.groupby(by)[col].transform(lambda x: x - x.median())
    elif strategy == 'zscore':
        scores = df.groupby(by)[col].transform(lambda x: zscore(x))
    else:
        scores = df[col]
    return scores


def debias(df_all, by='Reviewer Employee ID', col="BERT Sentiment Score", strategy='mean', **kwargs):
    imputation_options = {
        'impute': True,
        'cutoff': 5,
        'by': "Reviewer Department",
    }
    imputation_options.update(kwargs)

    logger.info(f"Debiasing scores using params {imputation_options}")

    # Set debiased score to NA for any where columns 'by' or 'col' are NA
    missing_idx = df_all.loc[df_all[[by, col]].isna().any(axis=1)].index
    df_nas = df_all.loc[missing_idx].copy()
    df_debias = df_all.loc[~df_all.index.isin(missing_idx)].copy()
    assert df_nas.shape[0] + df_debias.shape[0] == df_all.shape[0], \
        f"Number of reviews with missing columns in {[by, col]} do not sum up to total number of reviews: Expected " \
        f"{df_all.shape[0]}, got {df_nas.shape[0]} + {df_debias.shape[0]} instead"

    scores = _debias(df_debias, by, col, strategy)

    if imputation_options['impute']:
        is_sparse = df_debias.groupby(by)[col].transform(lambda x: x.count() < imputation_options['cutoff'])
        sparse_scores = _debias(df_debias, imputation_options['by'], col, strategy)
        scores.loc[is_sparse] = sparse_scores.loc[is_sparse]

    scores_scaled = min_max_scale(scores.to_numpy())
    df_debias[f'{col} (Debiased)'] = scores_scaled
    df_nas[f'{col} (Debiased)'] = np.nan

    df_all_normalized = pd.concat([df_debias, df_nas], axis=0)

    return df_all_normalized


if __name__ == "__main__":
    df_input = pd.read_csv(f"{BASE_DIR}data/derived/Sentiment Analysis HR {VERSION_NUMBER}.csv")

    df_reviews_text, df_reviews_non_text = split_reviews(df_input)
    df_reviews_text = debias(df_reviews_text, col="BERT Sentiment Score")
    df_reviews_processed = pd.concat([df_reviews_text, df_reviews_non_text], axis=0)

    df_reviews_processed.to_csv(f"{BASE_DIR}data/derived/Debiased Sentiment Analysis HR {VERSION_NUMBER}.csv",
                                index=False)
