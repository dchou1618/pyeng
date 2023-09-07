import os, sys
import pandas as pd
import re
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm

base_dir = '../'
sys.path.insert(1, base_dir)

import src.config as cfg
from src.normalize_scores import debias
import src.utils as ut

all_metric_cols = ['Rating', 'Rating (Debiased)',
                   'BERT Sentiment Score', 'BERT Sentiment Score (Debiased)'
                   ]
all_bin_cols = [f'{col} Bin' for col in all_metric_cols]
summary_cols = ['Employee ID', 'Business Unit',
                'Business Area', 'Corporate Title',
                'Functional Title', 'Revenue Generating?',
                'Department', 'City', 'Country',
                'Region']
all_modes = ['Detail', 'Summary', 'Backup']


def get_interaction_scores(df):
    """
    Get interaction score for each Review ID. Drop interaction scores so they're not used in average ratings
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    ut.logger.info(f"Getting interaction scores for {df.shape[0]} reviews.")

    df_interaction_level = df.loc[df['Question Text'].str.upper().str.contains("LEVEL OF INTERACTION|"
                                                                               "NIVEAU D’INTERACTION")]
    df_interaction_level = df_interaction_level[df_interaction_level["Response Text"] != ""]
    df_interaction_level = df_interaction_level.drop_duplicates(subset=["Review ID"])
    df_interaction_level = df_interaction_level[["Review ID", 'Response Text']]

    levels_of_interaction_map = {'Low': 1,
                                 'Medium': 2,
                                 'High': 3,
                                 'Faible': 1,
                                 'Modéré': 2,
                                 'Elevé': 3
                                 }
    missing_levels = df_interaction_level.loc[
        ~df_interaction_level['Response Text'].isin(levels_of_interaction_map.keys()), 'Response Text'].unique()
    ut.logger.warning(f"Missing levels of interaction in map: {missing_levels} - now setting those to NA")
    levels_of_interaction_map.update({level_of_interaction: np.nan for level_of_interaction in missing_levels})
    df_interaction_level['Interaction Score'] = df_interaction_level['Response Text'].replace(levels_of_interaction_map)
    df.drop_duplicates(subset=['Review ID', 'Response ID'], inplace=True)

    # Ignore levels of interaction when averaging ratings
    df = pd.merge(df,
                  df_interaction_level[['Review ID', 'Interaction Score']], on="Review ID", how="left")
    df['Interaction Score'] = df['Interaction Score'].astype(float)

    return df


def bin_cols(df, col, **kwargs):
    options = {
        'bins': [-1, -.6, -.2, .2, .6, 1],
        'labels': [1, 2, 3, 4, 5]
    }
    options.update(kwargs)

    ut.logger.info(f"Binning columns with params {options}")

    binned_values = pd.cut(df[col], bins=options['bins'], labels=options['labels'])
    return binned_values


def plot_dist(df, col="BERT Sentiment Score", fp=f"{cfg.BASE_DIR}data/figs/{cfg.VERSION_NUMBER}"):
    """
    Plot Distribution of Scores / Ratings
    :param df: pd.DataFrame
    :param col: str. column to plot
    :param fp: str. file path
    :return: None
    """
    ut.logger.info(f"Plotting distribution with column {col}")

    if not os.path.exists(fp):
        os.makedirs(fp)

    plt.figure(figsize=(14, 12))
    df[col].plot.kde()
    plt.xlabel(col)
    plt.ylabel("Probability Distribution")
    plt.savefig(f"""{fp}/{col} Distribution.png""")


def process_ratings(df, rating_col, min_rating, max_rating):
    """
    Set rating_col of [min_rating, max_rating] to NA and set review texts of NA manager/career sponsor assessments to NA
    :param df: pd.DataFrame
    :param rating_col: col
    :param min_rating: int
    :param max_rating: int
    :return: pd.DataFrame
    """
    # Ensure all ratings are on a max_rating-point scale
    ut.logger.info(f"Setting all {rating_col} to be on a scale where 'Max {rating_col}' is set to {max_rating}")
    df[rating_col] = df[rating_col] / df[f'Max {rating_col}'] * max_rating
    df[f'Max {rating_col}'] = max_rating

    # Identify Review ID's where the Manager/Career Sponsor Assessment was new-to-role or NA
    ut.logger.info(f"Setting all metrics where Question is about Manager/Career Sponsor Assessment and rating is NA "
                   f"or new-to-role as NA")
    df_manager_evaluation = df.loc[df['Question Text'].str.upper().str.contains("MANAGER\/CAREER SPONSOR ASSESSMENT|"
                                                                                "EVALUATION MANAGER")]
    no_manager_eval = df_manager_evaluation.loc[(df_manager_evaluation[rating_col] < min_rating) |
                                                (df_manager_evaluation[rating_col] > max_rating), 'Review ID'].unique()

    # Get corresponding Manager/Career Sponsor text questions and set the metric/bin columns to NA for those
    manager_eval_text_questions = ["Please provide examples that inform your rating.",
                                   "Donnez des exemples permettant d’illustrer votre évaluation."]
    df.loc[(df['Question Text'].isin(manager_eval_text_questions)) &
           (df['Review ID'].isin(no_manager_eval)),
           [col for col in all_metric_cols + all_bin_cols if col in df.columns]] = np.nan

    # Ignore ratings outside of bounds
    ut.logger.info(f"Setting all {rating_col} outside of [{min_rating}, {max_rating}] as NA")
    df.loc[(df[rating_col] < min_rating) | (df[rating_col] > max_rating), 'Rating'] = np.nan

    return df


def process_reviews(sentiment_filepath, rating_col, min_rating, max_rating):
    """
    Wrapper function to process reviews
    :param sentiment_filepath: str
    :param rating_col: str
    :param min_rating: int
    :param max_rating: int
    :return: pd.DataFrame
    """
    df_reviews = pd.read_hdf(sentiment_filepath)

    # Ensure all LDAG reviews are removed
    df_reviews = df_reviews.loc[df_reviews['Reviewee Department'] != 'Data Analytics Group'].copy()

    # Fill NA Response Text with ''
    df_reviews['Response Text'] = df_reviews['Response Text'].fillna('')
    df_reviews['Response ID'] = df_reviews['Response ID'].astype(str)
    df_reviews['Review ID'] = df_reviews['Review ID'].astype(str)

    # Map Reviewer roles to either ['360 Reviewer', 'Manager', 'Self']
    reviewer_role_map = {'360REV': '360 Reviewer',
                         '360CORP': '360 Reviewer',
                         'M': 'Manager',
                         'PEER': '360 Reviewer',
                         'PM/AN': '360 Reviewer',
                         'REV': '360 Reviewer',
                         'Employee': 'Self',
                         'E': 'Self',
                         'Self': 'Self',
                         'Manager': 'Manager',
                         'Contributor': "360 Reviewer",
                         '360 Reviewer': "360 Reviewer",
                         "360 ADMN": "360 Reviewer"}
    permitted_reviewer_roles = list(set(reviewer_role_map.values()))
    ut.logger.info(f"Mapping all reviewer roles to {permitted_reviewer_roles}")
    for role, relabeled_role in tqdm(reviewer_role_map.items()):
        df_reviews.loc[df_reviews['Reviewer Role'] == role, 'Reviewer Role'] = relabeled_role

    assert all([r in permitted_reviewer_roles for r in df_reviews['Reviewer Role'].unique()]), \
        f"Got reviewer roles not in {permitted_reviewer_roles}: {df_reviews['Reviewer Role'].unique()}"

    # Ensure non-text questions are set to NA sentiment
    df_reviews.loc[df_reviews['Question Type'] != 'Text', [col for col in df_reviews.columns if 'BERT' in col]] = np.nan

    df_reviews = process_ratings(df_reviews, rating_col, min_rating, max_rating)

    # Create debiased columns in all_metric_cols
    for col in all_metric_cols:
        if col not in df_reviews.columns:
            ut.logger.warning(f"Column {col} not in reviews dataframe {list(df_reviews.columns)}")
        elif 'Debiased' in col:
            continue
        else:
            df_reviews = debias(df_reviews, col=col)

    # Then plot all columns in all_metric_cols
    for col in all_metric_cols:
        if col not in df_reviews.columns:
            ut.logger.warning(f"Column {col} not in reviews dataframe {list(df_reviews.columns)}")
        else:
            plot_dist(df_reviews, col=col)

    # Create binned columns
    for col in all_metric_cols:
        if col not in df_reviews.columns:
            ut.logger.warning(f"Column {col} not in reviews dataframe {list(df_reviews.columns)}")
            continue
        elif col == 'Rating':
            df_reviews[f'{col} Bin'] = df_reviews[col]
        elif col == 'BERT Sentiment Score':
            df_reviews[f'{col} Bin'] = pd.cut(df_reviews[col], bins=np.arange(min_rating, max_rating + 1, 1),
                                              right=False)
        elif 'Debiased' in col:
            df_reviews[f'{col} Bin'] = bin_cols(df_reviews, col)
        else:
            raise Exception(f"Need to add details on how to bin column {col}. No column '{col} Bin' will be created")

    df_reviews = get_interaction_scores(df_reviews)

    return df_reviews


def aggregate_review_scores(df_raw, metrics, by="Reviewee", summary=False):
    """
    Pivot df so index are by Employee/Question ID if summary. Otherwise, index is just by Employee ID. Columns are
    "Reviewer Role" and values are metrics, aggregated by mean
    :param df_raw: pd.DataFrame
    :param metrics: list
    :param by: str
    :param summary: bool
    :return: pd.DataFrame
    """
    index = [f'{by} Employee ID', 'Question Text'] if not summary else [f'{by} Employee ID']

    # weighted average function
    def weighted_mean(x):
        return ut.nanaverage(x, weights=df_raw.loc[x.index, "Interaction Score"])

    # Dictionary of {reviewer role : aggregating function} - order here is maintained in output
    reviewer_roles_agg = {}
    if by == 'Reviewee':
        reviewer_roles_agg.update({
            'Manager': ['mean']
        })
    elif by == 'Reviewer':
        reviewer_roles_agg.update({
            'Manager': ['mean', 'std', 'count']
        })
    else:
        raise Exception(f"Must aggregate by a reviewer role in ['Reviewee', 'Reviewer'] - got {by} instead")

    reviewer_roles_agg.update({
        'Self': ['mean'],
        '360 Reviewer': [weighted_mean, 'mean', 'std', 'count']
    })

    df_agg = []
    output_bin_col_names = []  # Keep track of bin column names in output dataframe so we can fill NA (outer merge)
    # For each reviewer_role, group by agg_funcs
    for reviewer_role, agg_funcs in reviewer_roles_agg.items():
        df = df_raw.loc[df_raw['Reviewer Role'] == reviewer_role].groupby(index)[metrics].agg(agg_funcs)
        df = df.rename({col: f'{reviewer_role} - {col.upper()}' for col in df.columns.get_level_values(1).unique()},
                       axis=1, level=1)

        # Create binned counts for 360 Reviews + Reviewer-Manager
        if len(agg_funcs) > 1:
            df_bin = []  # Dataframe for each bin in all_bin_cols
            for bin_col in all_bin_cols:
                if bin_col not in df_raw.columns:
                    continue
                df_bin_col = (df_raw.loc[df_raw['Reviewer Role'] == reviewer_role,
                                         index + [bin_col, 'Question ID']]
                    .pivot_table(index=index, columns=bin_col, aggfunc="count")
                    .rename({
                    'Question ID': bin_col
                }, level=0, axis=1))
                df_bin_col = df_bin_col.rename({col: f'{reviewer_role} - {col}' for col in
                                                df_bin_col.columns.get_level_values(1).unique()},
                                               axis=1, level=1)
                df_bin_col.columns.names = [None, None]
                df_bin += [df_bin_col]
                output_bin_col_names += list(df_bin_col.columns.values)

            df_bin = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), df_bin)
            df = pd.merge(left=df, right=df_bin, left_index=True, right_index=True, how='outer')

        df_agg += [df]
    df_agg = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), df_agg)

    if len(output_bin_col_names) > 0:
        df_agg[output_bin_col_names] = df_agg[output_bin_col_names].replace(to_replace=0, value=np.nan)

    # Output order should be Rating, Rating Bin, Sentiment, Debiased Sentiment, Sentiment Bin
    output_cols = []
    all_cols_in_agg = list(df_agg.columns.get_level_values(0).unique())
    for col in metrics:
        if col not in all_cols_in_agg:
            ut.logger.warning(f"Column {col} not in reviews dataframe {all_cols_in_agg}")
            continue
        else:
            output_cols += [col]
            bin_col = f'{col} Bin'
            if bin_col not in all_cols_in_agg:
                ut.logger.warning(f"Column {bin_col} not in reviews dataframe {all_cols_in_agg}")
                continue
            else:
                output_cols += [bin_col]
    df_agg = df_agg[output_cols].copy()

    return df_agg


def get_employee_info(df, by="Reviewee", **kwargs):
    """
    Select LAZ employee data to later merge with reviews. Remove duplicates using by Employee ID
    :param df: pd.DataFrame
    :param by: str
    :return: pd.DataFrame
    """
    options = {
        'columns': summary_cols
    }
    options.update(kwargs)

    cols = [f'{by} {col}' for col in options['columns']]
    df_employee = df[cols].drop_duplicates(subset=[f'{by} Employee ID'])
    df_employee = df_employee.set_index(f'{by} Employee ID')
    df_employee.columns = [re.sub(by, "", col).strip() for col in df_employee.columns]
    df_employee.columns = pd.MultiIndex.from_product([[by], df_employee.columns])

    return df_employee


def get_backup_scores(df_raw, by='Reviewee'):
    """
    Generate backup tab of review question/response texts
    :param df_raw: pd.DataFrame
    :param by: str
    :return: pd.DataFrame
    """
    backup_cols = [f'{by} Employee ID', 'Review ID', 'Question ID', 'Response ID',
                   'Question Text', 'Reviewer Role']
    metric_cols = [col for col in all_metric_cols if col in df_raw.columns]
    bin_cols = [col for col in all_bin_cols if col in df_raw.columns]

    df_backup = df_raw.sort_values(backup_cols)[backup_cols + ['Question Type', 'Response Text'] +
                                                metric_cols + bin_cols]
    df_backup = df_backup.set_index(f'{by} Employee ID')
    df_backup.columns = pd.MultiIndex.from_product([['Question'], df_backup.columns])

    return df_backup


def generate_review_tables(df_raw, metrics, by="Reviewee", mode='Detail'):
    """
    Function that merges aggregated review/employee data in specific mode
    :param df_raw: pd.DataFrame
    :param metrics: list
    :param by: str
    :param mode: str
    :return: pd.DataFrame
    """
    df_employee_info = get_employee_info(df_raw, by)

    if mode not in all_modes:
        raise Exception(f"Mode must be in {all_modes}, got {mode} instead.")
    elif mode == 'Backup':
        df_backup = get_backup_scores(df_raw, by)
        df_backup = pd.merge(left=df_employee_info, right=df_backup, left_index=True, right_index=True, how='outer')
        return df_backup
    elif mode == 'Summary':
        summary = True
    else:
        summary = False

    df_agg_scores = aggregate_review_scores(df_raw, metrics, by, summary)
    df_proc = pd.merge(left=df_employee_info, right=df_agg_scores,
                       left_index=True, right_index=True, how='left')
    return df_proc


def write_review_tables(df_reviews, output_filepath, output_dict):
    """
    Generate and save all review tables using output_dict arguments on df_reviews to Excel file output_filepath
    :param df_reviews: pd.DataFrame
    :param output_filepath: str
    :param output_dict: dict. Keys are names for each tab in Excel output
    :return: None
    """
    writer = pd.ExcelWriter(output_filepath)
    metrics = [col for col in all_metric_cols if col in df_reviews.columns]

    for tab_name, output_options in output_dict.items():
        ut.logger.info(f"Saving {output_options} to {tab_name} in {output_filepath}")
        df = generate_review_tables(df_reviews, metrics, by=output_options['by'], mode=output_options['mode'])

        # To ensure we only have single index for Excel sorting/filtering
        df = df.reset_index(drop=False).set_index(f"{output_options['by']} Employee ID")
        if 'Question Text' in df.columns:
            df = df.rename({
                '': 'Question Text'
            }, level=1, axis=1)
        # Max tab name length is 31
        df.to_excel(writer, sheet_name=tab_name[0:31])

    writer.save()


def process_and_write_summary_pipeline():
    df_reviews_processed = process_reviews(f'{base_dir}cache.h5',
                                           rating_col='Rating', min_rating=1, max_rating=5)

    start_year, end_year = int(df_reviews_processed['Review Cycle Period'].min()), \
                           int(df_reviews_processed['Review Cycle Period'].max())


    for year in tqdm(np.arange(start_year, end_year + 1, 1)):
        write_review_tables(df_reviews_processed.loc[df_reviews_processed['Review Cycle Period'] == year],
                            f'{base_dir}output/{cfg.VERSION_NUMBER}/{year} Summary Analysis HR {cfg.VERSION_NUMBER}.xlsx',
                            output_dict={
                                'Reviewee Summary': {
                                    'by': 'Reviewee',
                                    'mode': 'Summary'
                                },
                                'Reviewee Detail': {
                                    'by': 'Reviewee',
                                    'mode': 'Detail'
                                },
                                'Reviewee Backup': {
                                    'by': 'Reviewee',
                                    'mode': 'Backup'
                                },
                                'Reviewer Summary': {
                                    'by': 'Reviewer',
                                    'mode': 'Summary'
                                },
                                'Reviewer Detail': {
                                    'by': 'Reviewer',
                                    'mode': 'Detail'
                                },
                                'Reviewer Backup': {
                                    'by': 'Reviewer',
                                    'mode': 'Backup'
                                }
                            }
                            )


if __name__ == "__main__":
    process_and_write_summary_pipeline()
