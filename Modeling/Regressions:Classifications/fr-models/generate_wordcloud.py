import os
import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from tqdm import tqdm
import nltk
import string

from nltk.corpus import stopwords

base_dir = '../'
sys.path.insert(1, base_dir)

from src.config import LAZ_CM, VERSION_NUMBER, BASE_DIR
from src.utils import split_reviews, logger

nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(stopwords.words('french'))
stop_words.extend(list(string.ascii_letters))
stop_words.extend(['tr', 'dan'])


def concat_text_by_group(df, by, text_col='Processed Response Text'):
    df_text_by_grp = (df.groupby(by)[text_col]
                      .apply(lambda x: ' '.join(x)))
    groups = df_text_by_grp.index.tolist()
    text_list = df_text_by_grp.tolist()
    return groups, text_list


def generate_wordcloud(text, fn, fp=f"{BASE_DIR}data/figs/{VERSION_NUMBER}"):
    # make directory if it doesn't exist
    if not os.path.exists(fp):
        os.makedirs(fp)

    fig_filepath = f"""{fp}/{fn}.png"""

    if not os.path.exists(fig_filepath):
        # generate word cloud
        wordcloud = WordCloud(
            background_color='white',
            width=2400,
            height=2000,
            colormap=LAZ_CM,
            stopwords=stop_words
        ).generate(text)
        plt.figure(figsize=(14, 12))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(fig_filepath)
        plt.clf()  # Clears current figure to avoid runtime warning that too many figures have been opened


def generate_wordclouds(df, by, text_col='Processed Response Text'):
    if df.empty:
        logger.warning("Got no text reviews, no wordclouds to generate")
        return pd.DataFrame()
    else:
        logger.info(f"Creating wordclouds for {df.shape[0]} reviews")

    groups, text_list = concat_text_by_group(df, by, text_col)
    for group, text in tqdm(zip(groups, text_list), total=len(text_list)):
        group = [str(col) for col in group]
        group_col_names = ".".join(group)
        fn = f"wordcloud_{group_col_names}"
        generate_wordcloud(text, fn)


def wordclouds_pipeline(df):
    df_text, df_non_text = split_reviews(df)
    df_text.dropna(subset=['Processed Response Text'], inplace=True)
    generate_wordclouds(df_text, by=['Reviewee Business Unit', 'Question ID'], text_col='Processed Response Text')


if __name__ == "__main__":
    df_all = pd.read_csv(f'{BASE_DIR}data/derived/Preprocessed HR incl Topic Counts {VERSION_NUMBER}.csv')
    wordclouds_pipeline(df_all)
