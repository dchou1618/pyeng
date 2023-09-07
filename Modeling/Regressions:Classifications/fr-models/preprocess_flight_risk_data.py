import codecs
import collections
import functools
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prefect import flow, task
from prefect.tasks import task_input_hash
from tqdm import tqdm

tqdm.pandas(desc='Progress Bar')


########################
# HR Parsing Functions #
########################

def datetime_to_str_date(x):
    return str(x).split()[0]


def extract_root_fpath(root_name, current_working_dir=os.path.dirname(__file__)):
    """
    @requires that root_name only exists once in the current working directory.
    """
    segmented_dir = current_working_dir.split("/")
    idx = segmented_dir.index(root_name)
    return "/".join(segmented_dir[:(idx + 1)])


@task(name="Parse HR data", cache_key_fn=task_input_hash)
def parse_hr_data(df, year_col, department_map, region_map):
    df["Department"] = df["Department"].replace(department_map)

    df = df.rename({'YOS(based on Last Hire DT)': 'Years of Service',
                    "YOS": "Years of Service"}, axis=1)
    df.loc[df["Years of Service"] < 0, "Years of Service"] = 0
    df["Years of Service (bin)"] = (
        pd.cut(
            df["Years of Service"],
            bins=[-1, 1, 2, 5, 10, np.infty],
            labels=["0-1 Years", "1-2 Years", "2-5 Years", "5-10 Years", "10+ Years"],
        )
        .astype(str)
        .replace({"nan": np.nan})
    )

    df["Bus Unit"] = df["Bus Unit"].replace(
        {"Private - LCM": "Asset Management", "Private - LAI": "Asset Management"}
    )

    missing_countries = df.loc[
        ~df["Country"].isin(list(region_map.keys())), "Country"
    ].unique()
    if len(missing_countries) > 0:
        raise Exception(f"Missing region map for following: {missing_countries}")
    df["Region"] = df["Country"].replace(region_map)
    if type(df[year_col].iloc[0]) != pd._libs.tslibs.timestamps.Timestamp:
        df[year_col] = df[year_col].apply(pd.Timestamp)
    df["Year"] = df[year_col].dt.year.astype("Int64", errors="ignore")
    # Only 1 datapoint in 2012, 0 datapoints in 2013
    df = df.loc[df["Year"] >= 2014].copy()

    return df


#################################
# HR Feature Creation Functions #
#################################

@task(name="Estimate the last promotion date of an employee", cache_key_fn=task_input_hash)
def get_last_promotion_date_df(df, promotions_df):
    last_promotion_date_col = []
    for i, row in df.iterrows():
        entries = promotions_df[(promotions_df["ID"] == row["ID"]) &
                                (promotions_df["Eff Date"].apply(lambda x: x.year)
                                 < row["Year"])]
        if len(entries) == 0:
            if not pd.isna(row["Term Date"]):
                if row["Year"] == row["Term Date"].year:
                    hire_dates = df[(df["ID"] == row["ID"]) &
                                    (df["Hire Date"] < row["Term Date"])].sort_values("Hire Date",
                                                                                      ascending=False)
                else:
                    hire_dates = df[(df["ID"] == row["ID"]) &
                                    (df["Hire Date"] < datetime(row["Year"], 12, 31))].sort_values("Hire Date",
                                                                                                   ascending=False)
                if len(hire_dates) > 0:
                    last_promotion_date_col.append(hire_dates["Hire Date"].iloc[0])
                else:
                    last_promotion_date_col.append(row["Hire Date"])
            else:
                if row["Year"] != datetime.today().year:
                    hire_dates = df[(df["ID"] == row["ID"]) &
                                    (df["Hire Date"] < datetime(row["Year"], 1, 1))].sort_values("Hire Date",
                                                                                                 ascending=False)
                    if len(hire_dates) > 0:
                        last_promotion_date_col.append(hire_dates["Hire Date"].iloc[0])
                    else:
                        last_promotion_date_col.append(row["Hire Date"])
                else:
                    last_promotion_date_col.append(row["Hire Date"])
        else:
            last_promotion_date_col.append(entries["Eff Date"].iloc[0])
    return last_promotion_date_col


def get_leaver_age(row):
    if row["Year"] == row["Term Date"].year:
        diff = row["Term Date"] - row["DOB"]
    else:
        diff = datetime(row["Year"], 1, 1) - row["DOB"]
    return diff.days / 365.2425


def get_stayer_age(row):
    if row["Year"] == datetime.today().year:
        diff = datetime.today() - row["DOB"]
    else:
        diff = datetime(row["Year"], 1, 1) - row["DOB"]
    return diff.days / 365.2425


def map_age(row):
    return get_stayer_age(row) if row["Attrition"] == 0 else get_leaver_age(row)


def get_yrs_in_title(row):
    if not pd.isna(row["Term Date"]):
        if row["Year"] == row["Term Date"].year:
            last_date_after_promotion = row["Term Date"]
        else:
            assert row["Year"] < row["Term Date"].year, "Year of record not before Termination"
            last_date_after_promotion = datetime(row["Year"], 12, 31)
    else:
        if row["Year"] == datetime.today().year:
            last_date_after_promotion = datetime.today()
        else:
            last_date_after_promotion = datetime(row["Year"], 12, 31)
    days_in_title = (last_date_after_promotion - row["Last Promotion Date"]).days
    return days_in_title / 365.2425


@task(name="Add Compensation Percentage Change", cache_key_fn=task_input_hash)
def compensation_perc_change(df_all):
    df_all["YOY TC % Chg T-1"] = df_all["YOY TC % Chg T-1"].replace("-", float('-inf')).apply(float)
    df_all["YOY TC % Chg T-2"] = df_all["YOY TC % Chg T-2"].replace("-", float('-inf')).apply(float)
    for col in ["T-1", "T-2"]:
        df_all["YOY TC % Chg.1 " + col] = df_all["YOY TC % Chg " + col].apply(modify_yoy_tc_chg)

    df_all = df_all.drop(columns=["Corporate Group"])

    for col in ["T-1", "T-2"]:
        df_all = df_all.rename(columns={"YOY TC % Chg.1 " + col: "YOY TC % Chg (bin) " + col})

    col_mapping = dict()
    for col in df_all.columns:
        if ".1" in col:
            col_mapping[col] = col.replace(".1", "")
    df_all = df_all.rename(columns=col_mapping)
    for col in ["T-1", "T-2"]:
        df_all["YOY TC % Chg " + col] = df_all["YOY TC % Chg " + col].replace(float("-inf"), np.nan)

    return df_all


def get_hc_changes(row, hc_dept_region):
    available_hcs = \
        hc_dept_region[(hc_dept_region["Region"] == row["Region"]) & \
                       (hc_dept_region["Department"] == row["Department"]) & \
                       (hc_dept_region["Year"] <= row["Year"])].sort_values("Year")
    hcs = list(available_hcs["HC by Department, Region"])
    if len(hcs) < 2:
        return np.nan, np.nan
    else:
        all_changes = [hcs[i + 1] - hcs[i] for i in range(len(hcs) - 1)]
        latest_change = all_changes[-1]
        # latest and median change from the past years
        return latest_change, all_changes


@task(name="Adding Headcount Changes Variables", cache_key_fn=task_input_hash)
def headcount_changes(df_all):
    # Headcount Change Features
    def f(x):
        if x.count().iloc[0] <= 1:
            return pd.Series(np.nan, index=["Attrition"])
        return x.count()

    hc_by_region = df_all.groupby(["Year", "Region"], as_index=False).count()
    hc_dept_region = df_all.groupby(["Year", "Region", "Department"], as_index=False)[["Attrition"]] \
        .apply(f).rename(columns={"Attrition": "HC by Department, Region"})

    hc_dept_region["HC by Department, Region"] = hc_dept_region.apply(lambda row:
                                                                      hc_by_region[
                                                                          (hc_by_region["Year"] == row["Year"]) & \
                                                                          (hc_by_region["Region"] == row["Region"])][
                                                                          "Attrition"].iloc[0] \
                                                                          if pd.isna(
                                                                          row["HC by Department, Region"]) else row[
                                                                          "HC by Department, Region"],
                                                                      axis=1)

    df_all[["HC Change 1 YR by Department, Region",
            "HC Changes by Department, Region List"]] = df_all.progress_apply(lambda row: get_hc_changes(row,
                                                                                                         hc_dept_region),
                                                                              axis=1,
                                                                              result_type="expand")
    return df_all


def get_emp_group_curr(row, emp_group_dept_relative, emp_group_region_relative):
    emp_group_curr = \
        emp_group_dept_relative[(emp_group_dept_relative["Region"] == row["Region"]) & \
                                (emp_group_dept_relative["Department"] == row["Department"]) & \
                                (emp_group_dept_relative["Year"] <= row["Year"])].sort_values("Year")
    emp_group_curr_2 = \
        emp_group_region_relative[(emp_group_region_relative["Region"] == row["Region"]) & \
                                  (emp_group_region_relative["Year"] <= row["Year"])].sort_values("Year")
    emp_group_curr_years = set(emp_group_curr["Year"])
    emp_group_curr_2 = \
        emp_group_curr_2[emp_group_curr_2["Year"].isin(emp_group_curr_years)]
    assert len(emp_group_curr) == len(emp_group_curr_2) and list(emp_group_curr["Year"]) == \
           list(emp_group_curr_2["Year"]) \
           and list(emp_group_curr["Year"]) == sorted(list(emp_group_curr["Year"])), \
        f"{emp_group_curr}, {emp_group_curr_2}"
    return emp_group_curr, emp_group_curr_2


def get_ratios(group, fallback_group):
    d = collections.defaultdict(int)
    for emp in group:
        d[emp] += 1
    d_fallback = collections.defaultdict(int)
    for emp in fallback_group:
        d_fallback[emp] += 1

    if d["Managing Director"] == 0:
        analyst_mds = d_fallback["Analyst"] / d_fallback["Managing Director"]
        associate_mds = d_fallback["Associate"] / d_fallback["Managing Director"]
    else:
        analyst_mds = d["Analyst"] / d["Managing Director"]
        associate_mds = d["Associate"] / d["Managing Director"]
    if (d["Managing Director"] + d["Director"]) == 0:
        analyst_mds_dirs = d_fallback["Analyst"] / (d_fallback["Managing Director"] + d_fallback["Director"])
        associate_mds_dirs = d_fallback["Associate"] / (d_fallback["Managing Director"] + d_fallback["Director"])
        analyst_associates_mds_dirs = (d_fallback["Analyst"] + d_fallback["Associate"]) / \
                                      (d_fallback["Managing Director"] + d_fallback["Director"])
    else:
        analyst_mds_dirs = d["Analyst"] / (d["Managing Director"] + d["Director"])
        associate_mds_dirs = d["Associate"] / (d["Managing Director"] + d["Director"])
        analyst_associates_mds_dirs = (d["Analyst"] + d["Associate"]) / \
                                      (d["Managing Director"] + d["Director"])
    # addition of Analysts/Associates and Analysts/(Associates+VPs)
    if d["Associate"] == 0:
        analyst_assoc = d_fallback["Analyst"] / d_fallback["Associate"]
    else:
        analyst_assoc = d["Analyst"] / d["Associate"]

    if d["Associate"] + d["Vice President"] == 0:
        analyst_assoc_vp = d_fallback["Analyst"] / (d_fallback["Associate"] + d_fallback["Vice President"])
    else:
        analyst_assoc_vp = d["Analyst"] / (d["Associate"] + d["Vice President"])

    return analyst_mds, associate_mds, analyst_mds_dirs, \
        associate_mds_dirs, analyst_associates_mds_dirs, analyst_assoc, analyst_assoc_vp


def get_hc_ratio(row, emp_group_dept, emp_group_region):
    emp_group_curr, emp_group_curr_2 = get_emp_group_curr(row, emp_group_dept, emp_group_region)

    fallback_title_groups = list(emp_group_curr_2["Corporate Title"])

    title_groups = list(emp_group_curr["Corporate Title"])
    ratios_per_group_over_yrs = [get_ratios(group, fallback_group) \
                                 for fallback_group, group in
                                 zip(fallback_title_groups, title_groups)]
    analyst_mds = [ratios[0] for ratios in ratios_per_group_over_yrs]
    associate_mds = [ratios[1] for ratios in ratios_per_group_over_yrs]
    analyst_mds_dirs = [ratios[2] for ratios in ratios_per_group_over_yrs]
    associate_mds_dirs = [ratios[3] for ratios in ratios_per_group_over_yrs]
    analyst_associates_mds_dirs = [ratios[4] for ratios in ratios_per_group_over_yrs]
    analyst_assocs = [ratios[5] for ratios in ratios_per_group_over_yrs]
    analyst_assoc_vps = [ratios[6] for ratios in ratios_per_group_over_yrs]
    # obtain ratios
    return analyst_mds, associate_mds, analyst_mds_dirs, \
        associate_mds_dirs, analyst_associates_mds_dirs, analyst_assocs, analyst_assoc_vps


@task(name="Adding headcount ratios", cache_key_fn=task_input_hash)
def headcount_ratios(df_all):
    df_all = df_all.reset_index(drop=True)
    emp_group_region = df_all \
        .groupby(["Year", "Region"], as_index=False).agg(list)[["ID", "Year",
                                                                "Region",
                                                                "Corporate Title",
                                                                "Attrition",
                                                                "Age", "Sex",
                                                                "360 Reviewer - MEDIAN T-1",
                                                                "360 Reviewer - MEAN T-1",
                                                                "Junior ID"]]
    emp_group_dept = df_all \
        .groupby(["Year", "Region", "Department"],
                 as_index=False).agg(list)[["ID", "Year", "Region", "Department", "Corporate Title", "Attrition",
                                            "Age", "Sex",
                                            "360 Reviewer - MEDIAN T-1",
                                            "360 Reviewer - MEAN T-1",
                                            "Junior ID"]]

    # Ratio of Headcounts Features
    firmwide_groups = df_all[["Year", "Corporate Title"]].groupby(["Year"], as_index=False).agg(list)

    firmwide_groups[["Firmwide Analysts/MDs",
                     "Firmwide Associates/MDs",
                     "Firmwide Analysts/MDs+Dirs",
                     "Firmwide Associates/MDs+Dirs",
                     "Firmwide Analysts+Associates/MDs+Dirs",
                     "Firmwide Analysts/Associates",
                     "Firmwide Analysts/Associates+VPs"]] = \
        firmwide_groups.apply(lambda row: get_ratios(row["Corporate Title"], row["Corporate Title"]),
                              result_type="expand", axis=1)
    df_all = pd.merge(df_all, firmwide_groups[["Year", "Firmwide Analysts/MDs",
                                               "Firmwide Associates/MDs",
                                               "Firmwide Analysts/MDs+Dirs",
                                               "Firmwide Associates/MDs+Dirs",
                                               "Firmwide Analysts+Associates/MDs+Dirs",
                                               "Firmwide Analysts/Associates",
                                               "Firmwide Analysts/Associates+VPs"]],
                      on=["Year"], how="left")
    df_all[["Historical HC Analysts/MDs",
            "Historical HC Associates/MDs",
            "Historical HC Analysts/MDs+Dirs",
            "Historical HC Associates/MDs+Dirs",
            "Historical HC Analysts+Associates/MDs+Dirs",
            "Historical Analysts/Associates",
            "Historical Analysts/Associates+VPs"]] = \
        df_all.progress_apply(lambda row: get_hc_ratio(row, emp_group_dept, emp_group_region), axis=1,
                              result_type="expand")

    for ratio_var in ["Historical HC Analysts/MDs",
                      "Historical HC Associates/MDs",
                      "Historical HC Analysts/MDs+Dirs",
                      "Historical HC Associates/MDs+Dirs",
                      "Historical HC Analysts+Associates/MDs+Dirs",
                      "Historical Analysts/Associates",
                      "Historical Analysts/Associates+VPs"]:
        df_all["Latest " + ratio_var.split()[-1]] = \
            df_all[ratio_var].apply(lambda lst: lst[-1] if len(lst) > 0 else np.nan)

    return df_all, emp_group_dept, emp_group_region


def prepare_reviews(df_reviews):
    df_reviews = df_reviews.loc[df_reviews['Reviewee Department'] != 'Data Analytics Group'].copy()
    # Fill NA Response Text with ''
    df_reviews['Response Text'] = df_reviews['Response Text'].fillna('')

    # Map Reviewer roles to either ['360 Reviewer', 'Manager', 'Self']
    reviewer_role_map = {'360REV': '360 Reviewer',
                         '360CORP': '360 Reviewer',
                         'M': 'Manager',
                         'PEER': '360 Reviewer',
                         'PM/AN': '360 Reviewer',
                         'REV': '360 Reviewer',
                         'Employee': 'Self',
                         'E': 'Self'}
    permitted_reviewer_roles = list(set(reviewer_role_map.values()))
    for role, relabeled_role in tqdm(reviewer_role_map.items()):
        df_reviews.loc[df_reviews['Reviewer Role'] == role, 'Reviewer Role'] = relabeled_role
    assert all([r in permitted_reviewer_roles for r in df_reviews['Reviewer Role'].unique()]), \
        f"Got reviewer roles not in {permitted_reviewer_roles}: {df_reviews['Reviewer Role'].unique()}"

    return df_reviews


def read_manager_data(root_path, pre2020_relpath, ye2021_relpath):
    df_reviews = pd.read_csv(f"{root_path}/{pre2020_relpath}")
    df_reviews = prepare_reviews(df_reviews)
    ye2021 = pd.read_csv(f"{root_path}/{ye2021_relpath}")
    pre2020 = df_reviews[df_reviews["Reviewer Role"] == "Manager"][["Reviewer Employee ID", "Reviewee Employee ID",
                                                                    "Review Cycle Period"]].dropna()
    ye2021 = ye2021[ye2021["Reviewer Role"] == "Manager"][["Reviewer Employee ID", "Reviewee Employee ID",
                                                           "Review Cycle Period"]].dropna()
    return pd.concat([pre2020, ye2021]).drop_duplicates()


@task(name="Adding Junior IDs and returning manager_ids for downstream functions", cache_key_fn=task_input_hash)
def manager_specific(df_all, root_path, pre2020_relpath, ye2021_relpath):
    manager_ids = read_manager_data(root_path, pre2020_relpath, ye2021_relpath)

    manager_ids = manager_ids.rename(columns={"Reviewer Employee ID": "ID",
                                              "Reviewee Employee ID": "Junior ID",
                                              "Review Cycle Period": "Year"}) \
        .groupby(["ID", "Year"], as_index=False) \
        .agg(list)[["ID", "Junior ID", "Year"]].dropna()
    manager_ids["Year"] = manager_ids["Year"].apply(lambda year: int(year + 1))
    df_all = pd.merge(df_all, manager_ids, how="left",
                      on=["ID", "Year"])

    return df_all


@task(name="Adding compensation-related features", cache_key_fn=task_input_hash)
def compensation_related_features(df_all):
    # Computing CAGR #
    df_all["YOY TC % Chg CAGR"] = df_all.apply(lambda row: np.nan \
        if pd.isna(row["YOY TC % Chg T-1"]) or pd.isna(row["YOY TC % Chg T-2"]) \
        else (1 + row["YOY TC % Chg T-1"]) * (1 + row["YOY TC % Chg T-2"]) - 1, axis=1)
    tier_mapping = dict()
    tier_i = 0
    for i in range(1, 6):
        for val in [str(i) + "-", i, str(i) + "+"]:
            tier_mapping[val] = tier_i
            tier_i += 1
    print("\n\n", df_all["Comp Tier T-1"].unique(), "\n\n")
    quartile_mapping = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "--": np.nan}
    df_all["Comp Tier Change T-1"] = df_all.apply(lambda row: np.nan \
        if pd.isna(row["Comp Tier T-1"]) or pd.isna(row["Comp Tier T-2"]) \
        else (tier_mapping[row["Comp Tier T-1"]] - tier_mapping[row["Comp Tier T-2"]]), axis=1)

    df_all["TC Mkt Data Quartile Change T-1"] = df_all.apply(lambda row: np.nan \
        if pd.isna(row["TC Mkt Data Quartile T-1"]) or pd.isna(row["TC Mkt Data Quartile T-2"]) \
        else (quartile_mapping[row["TC Mkt Data Quartile T-1"]] - quartile_mapping[row["TC Mkt Data Quartile T-2"]]),
                                                             axis=1)

    for col in ["T-1", "T-2"]:
        df_all["Comp Tier " + col] = df_all["Comp Tier " + col].apply(
            lambda x: tier_mapping[x] if x in tier_mapping else x)
        df_all["TC Mkt Data Quartile " + col] = df_all["TC Mkt Data Quartile " + col].apply(
            lambda x: quartile_mapping[x] if x in quartile_mapping else x)
    return df_all


def attrition_for_dept_region(x):
    if x.count().iloc[0] <= 1:
        return pd.Series(np.nan, index=["Attrition"])
    return x[x.Attrition == 1].count() / x.count()


def attrition_for_supv(x):
    return x[x.Attrition == 1].count() / x.count()


def headcount_for_region(x):
    if x.count().iloc[0] <= 1:
        return pd.Series(np.nan, index=["Attrition"])
    return x.count()


@task(name="Add group attrition features (derived)", cache_key_fn=task_input_hash)
def group_attrition(df_all):
    for col in ["T-1", "T-2"]:
        df_all["TC Mkt Data Quartile " + col] = \
            df_all["TC Mkt Data Quartile " + col].copy().replace(["-"], np.nan)

    hc_per_region_bus_area = df_all \
        .groupby(["Year", "Region", "Business Area"],
                 as_index=False)[["Attrition"]].apply(lambda x: x.count()) \
        .rename(columns={"Attrition": "HC by Business Area, Region"})
    df_all = pd.merge(df_all,
                      hc_per_region_bus_area,
                      on=["Year", "Region", "Business Area"],
                      how="left")

    rate_by_region = df_all \
        .groupby(["Year", "Region"], as_index=False)[["Attrition"]] \
        .apply(lambda x: x[x.Attrition == 1].count() / x.count())

    dept_region_per_yr = df_all \
        .groupby(["Year", "Region", "Department"],
                 as_index=False)[["Attrition"]] \
        .apply(attrition_for_dept_region)

    dept_region_per_yr["Attrition"] = dept_region_per_yr.apply(lambda row:
                                                               rate_by_region[(rate_by_region["Year"] == row["Year"]) & \
                                                                              (rate_by_region["Region"] == row[
                                                                                  "Region"])]["Attrition"].iloc[0] \
                                                                   if pd.isna(row["Attrition"]) else row["Attrition"],
                                                               axis=1)
    df_all = pd.merge(df_all,
                      dept_region_per_yr.rename(columns={"Attrition": "Turnover Rate by Department, Region"}),
                      on=["Year", "Region", "Department"],
                      how="left")

    hc_by_region = df_all \
        .groupby(["Year", "Region"], as_index=False)[["Attrition"]] \
        .apply(lambda x: x.count())

    hc_region_dept = df_all \
        .groupby(["Year", "Region", "Department"],
                 as_index=False)[["Attrition"]] \
        .apply(headcount_for_region)

    hc_region_dept["Attrition"] = hc_region_dept.apply(lambda row:
                                                       hc_by_region[(hc_by_region["Year"] == row["Year"]) & \
                                                                    (hc_by_region["Region"] == row["Region"])][
                                                           "Attrition"].iloc[0] \
                                                           if pd.isna(row["Attrition"]) else row["Attrition"],
                                                       axis=1)

    df_all = pd.merge(df_all,
                      hc_region_dept.rename(columns={"Attrition": "HC by Department, Region"}),
                      on=["Year", "Region", "Department"],
                      how="left")

    supv_uniq_employees = df_all.copy()
    supv_uniq_employees["Supv ID"] = supv_uniq_employees["Supv ID"].replace(np.nan, "None")
    assert all(not pd.isna(emp) for emp in supv_uniq_employees["Supv ID"].values), \
        "There's at least one NaN Supervisor ID"
    turnover_per_supv = supv_uniq_employees \
        .groupby(["Year", "Supv ID"], as_index=False)[["Attrition"]] \
        .apply(attrition_for_supv)

    df_all["Supv ID"] = df_all["Supv ID"].replace(np.nan, "None")
    df_all = pd.merge(df_all,
                      turnover_per_supv.rename(columns={"Attrition": "Turnover Rate by Supervisor"}),
                      on=["Year", "Supv ID"], how="left")

    hc_per_supv = supv_uniq_employees \
        .groupby(["Year", "Supv ID"], as_index=False)[["Attrition"]] \
        .apply(lambda x: x.count())
    df_all = pd.merge(df_all,
                      hc_per_supv.rename(columns={"Attrition": "HC by Supervisor"}),
                      on=["Year", "Supv ID"], how="left")

    emp_supv_groups = supv_uniq_employees.groupby(["Supv ID", "Year"], as_index=False).agg(list)
    return df_all, emp_supv_groups


def get_relative_ordering_list(latest_group, title_ordering, row_title_num, relative_to_junior):
    if relative_to_junior:
        idx_title = [(title_ordering[val], i) for i, val in enumerate(latest_group["Corporate Title"]) \
                     if val not in {"Not Applicable", "Senior Advisor"} and title_ordering[val] < row_title_num]
    else:
        idx_title = [(title_ordering[val], i) for i, val in enumerate(latest_group["Corporate Title"]) \
                     if val not in {"Not Applicable", "Senior Advisor"} and title_ordering[val] > row_title_num]
    return idx_title


def relative_turnover_helper(title_ordering, title_groups, fallback_title_groups, row_title_num, relative_to_junior):
    latest_dept_group = title_groups.iloc[-1]
    latest_dept_fallback_group = fallback_title_groups.iloc[-1]
    idx_title = \
        get_relative_ordering_list(latest_dept_group, title_ordering,
                                   row_title_num, relative_to_junior)

    if len(idx_title) == 0:
        idx_title = \
            get_relative_ordering_list(latest_dept_fallback_group, title_ordering,
                                       row_title_num, relative_to_junior)
        attrition_lst = [latest_dept_fallback_group["Attrition"][idx] for _, idx in idx_title]
    else:
        assert not any(idx >= len(latest_dept_group["Attrition"]) for _, idx in idx_title), \
            "Index in turnover list exceeds full length of employees at same title in latest department group"
        attrition_lst = [latest_dept_group["Attrition"][idx] for _, idx in idx_title]
    # no one is below in title to the individual.
    if len(attrition_lst) == 0:
        return np.nan
    return len([val for val in attrition_lst if val == 1]) / len(attrition_lst)


def get_relative_turnover(row, title_ordering, emp_group_dept, emp_group_region, relative_to_junior=True):
    """
    get_relative_turnover
    :param row: pd.Series
    :param title_ordering: dict
    :param emp_group_dept: pd.DataFrame
    :param emp_group_region: pd.DataFrame
    :param relative_to_junior: str
    :return: float
    """
    emp_group_curr, emp_group_curr_2 = get_emp_group_curr(row, emp_group_dept, emp_group_region)
    if row["Corporate Title"] in {"Senior Advisor", "Not Applicable"}:
        return np.nan
    return relative_turnover_helper(title_ordering, emp_group_curr, emp_group_curr_2,
                                    title_ordering[row["Corporate Title"]],
                                    relative_to_junior)


@task(name="Adding relative turnover rates - those junior or senior to an employee", cache_key_fn=task_input_hash)
def relative_attrition(df_all, emp_group_dept, emp_group_region):
    title_ordering = {title: i for i, title in enumerate(["Analyst",
                                                          "Associate",
                                                          "Vice President",
                                                          "Senior Vice President",
                                                          "Director", "Managing Director"])}

    # create feature in turnover of those junior to them in same group.

    df_all_group_junior = df_all.progress_apply(lambda row: get_relative_turnover(row, title_ordering, emp_group_dept,
                                                                                  emp_group_region, True), axis=1,
                                                result_type="expand")

    # create feature in turnover of those senior to them in same group.

    df_all_group_senior = df_all.progress_apply(lambda row: get_relative_turnover(row, title_ordering,
                                                                                  emp_group_dept,
                                                                                  emp_group_region,
                                                                                  False),
                                                axis=1,
                                                result_type="expand")

    df_all["Turnover in Group Among Juniors"] = df_all_group_junior
    df_all["Turnover in Group Among Seniors"] = df_all_group_senior
    return df_all


def age_relative_to_title_helper(dept_level_df, region_level_df, row_age, row_title):
    '''
    requires: the input dataframes are already employees that are within
    the same group as the employee at "row"
    '''
    latest_dept_level = dept_level_df.iloc[-1]
    latest_region_level = region_level_df.iloc[-1]
    idx_title = [i for i, val in enumerate(latest_dept_level["Corporate Title"]) if val == row_title]

    if len(idx_title) < 2:
        idx_title = [i for i, val in enumerate(latest_region_level["Corporate Title"]) if val == row_title]
        region_ages = [latest_region_level["Age"][idx] for idx in idx_title]
        if len(region_ages) == 0:
            return np.nan

        output = (row_age - np.nanmean(region_ages)) / np.std(region_ages)
    else:
        dept_ages = [latest_dept_level["Age"][idx] for idx in idx_title]
        if len(dept_ages) == 0:
            return np.nan
        output = (row_age - np.nanmean(dept_ages)) / np.std(dept_ages)
    return output


def age_relative_to_title(row, emp_group_dept, emp_group_region):
    emp_group_curr, emp_group_curr_2 = get_emp_group_curr(row, emp_group_dept, emp_group_region)

    return age_relative_to_title_helper(emp_group_curr, emp_group_curr_2,
                                        row["Age"], row["Corporate Title"])


@task(name="Adding z-score of age relative to same title as employee", cache_key_fn=task_input_hash)
def compute_relative_age_to_title(df_all, emp_group_dept, emp_group_region):
    df_all["Age"] = df_all.apply(map_age, axis=1)
    relative_age_to_title = df_all.progress_apply(lambda row: age_relative_to_title(row, emp_group_dept,
                                                                                    emp_group_region),
                                                  axis=1,
                                                  result_type="expand")
    df_all["Age Relative to Title"] = relative_age_to_title
    return df_all


def get_prop_women(lst):
    return len([val for val in lst if val == "Female"]) / len(lst)


# get % women in supervisor's group
def perc_women_under_supervisor(row, emp_supv_groups):
    gender_in_groups = emp_supv_groups[(emp_supv_groups["Year"] <= row["Year"]) & \
                                       (emp_supv_groups["Supv ID"] == row["Supv ID"])]
    gender_in_groups = gender_in_groups.sort_values("Year")
    # from earliest to most recent proportion of women in group
    proportion_of_women_by_supervisor = gender_in_groups["Sex"].apply(get_prop_women).values
    return list(proportion_of_women_by_supervisor)


# get % women in department
def perc_women_in_dept(row, emp_group_dept, emp_group_region):
    emp_group_curr, _ = get_emp_group_curr(row, emp_group_dept, emp_group_region)
    # from earliest to latest proportion of women.
    gender_in_depts = emp_group_curr["Sex"].apply(get_prop_women)
    return list(gender_in_depts.values)


@task(name="Adding Percentage of Women and Percentage Change in Women", cache_key_fn=task_input_hash)
def compute_perc_women_factors(df_all, emp_supv_groups, emp_group_dept, emp_group_region):
    past_perc_women_dept = df_all.progress_apply(lambda row: perc_women_in_dept(row, emp_group_dept,
                                                                                emp_group_region), axis=1)
    past_perc_diff_women_dept = past_perc_women_dept.progress_apply(lambda x: [x[i + 1] - \
                                                                               x[i] for i in range(len(x) - 1)] if len(
        x) > 1 else np.nan)
    past_perc_women_supv = df_all.progress_apply(lambda row: perc_women_under_supervisor(row, emp_supv_groups), axis=1)
    past_perc_diff_women_supv = past_perc_women_supv.progress_apply(lambda x: [x[i + 1] - \
                                                                               x[i] for i in range(len(x) - 1)] if len(
        x) > 1 else np.nan)
    perc_women_column_names = ["Historical % Women by Supervisor",
                               "Historical % Change in Women by Supervisor",
                               "Historical % Women by Department",
                               "Historical % Change in Women by Department"]
    col_data = [past_perc_women_supv, past_perc_diff_women_supv,
                past_perc_women_dept, past_perc_diff_women_dept]
    for col_name, data in zip(perc_women_column_names, col_data):
        df_all[col_name] = data.values
    return df_all, perc_women_column_names


# we retrieve average number of people that report to their manager in a department
# we also record the average 360 review of managers in the department.
def get_avg_manager_attributes(group_latest):
    indices = []
    for i in range(len(group_latest["ID"])):
        curr_juniors_to_manager = group_latest["Junior ID"][i]
        if (type(curr_juniors_to_manager) == list) and (len(curr_juniors_to_manager) > 0):
            indices.append(i)
    reviews = [group_latest["360 Reviewer - MEAN T-1"][i] for i in indices]
    #     bin_reviews = [group_latest["360 Reviewer - MEAN T-1 (bin)"][i] for i in indices]
    if len(reviews) == 0:
        return np.nan, np.nan
    # if reviews are 0-length, this means no identifiable managers in group.
    junior_lsts = [len(group_latest["Junior ID"][i]) for i in indices]

    reviews = [x for x in reviews if not pd.isna(x)]
    #     bin_reviews = [float(x) for x in bin_reviews if not pd.isna(x)]
    if len(reviews) == 0:
        avg_mean_group_review, avg_bin_review = np.nan, np.nan
    else:
        avg_mean_group_review = sum(reviews) / len(reviews)
    avg_to_manager = sum(junior_lsts) / len(junior_lsts)
    return avg_to_manager, avg_mean_group_review


def get_leadership_aggregates(row, emp_group_dept, emp_group_region):
    emp_group_curr, emp_region_curr = get_emp_group_curr(row, emp_group_dept, emp_group_region)
    emp_group_curr = emp_group_curr.sort_values("Year")
    emp_region_curr = emp_region_curr.sort_values("Year")
    latest_emp_group = emp_group_curr.iloc[-1]
    latest_emp_region = emp_region_curr.iloc[-1]
    avg_to_manager, avg_mean_group_review = get_avg_manager_attributes(latest_emp_group)
    if pd.isnull(avg_to_manager):
        return get_avg_manager_attributes(latest_emp_region)
    return avg_to_manager, avg_mean_group_review


@task(name="Adding manager-specific feature(s) to data", cache_key_fn=task_input_hash)
def compute_manager_specific(df_all, emp_group_dept, emp_group_region):
    df_all[["Average Employees to Manager in Department, Region",
            "Average Mean Review of Managers in Department, Region"]] = \
        df_all.progress_apply(lambda row: get_leadership_aggregates(row, emp_group_dept, emp_group_region), axis=1,
                              result_type="expand")
    return df_all


def get_title_counts(titles):
    d = collections.defaultdict(int)
    for title in titles:
        d[title] += 1
    return d


def department_title_changes(row, emp_group_dept, emp_group_region):
    emp_group_curr, emp_group_curr_2 = get_emp_group_curr(row, emp_group_dept, emp_group_region)

    fallback_title_groups = list(emp_group_curr_2["Corporate Title"])
    title_groups = list(emp_group_curr["Corporate Title"])

    row_title = row["Corporate Title"]
    title_groups = [group[row_title] for group in title_groups]
    if any(group == 0 for group in title_groups):
        title_groups = [group[row_title] for group in fallback_title_groups]
    chgs = [title_groups[i + 1] - title_groups[i] for i in range(len(title_groups) - 1)]
    if len(chgs) == 0:
        return np.nan, np.nan
    return chgs[-1], chgs


@task(name="Adding title-related features to data", cache_key_fn=task_input_hash)
def title_related_features(df_all, emp_group_dept, emp_group_region):
    emp_group_dept["Corporate Title"] = \
        emp_group_dept.apply(lambda row: get_title_counts(row["Corporate Title"]), axis=1)
    emp_group_region["Corporate Title"] = \
        emp_group_region.apply(lambda row: get_title_counts(row["Corporate Title"]), axis=1)
    df_all[["HC Change 1 YR in Title by Department, Region",
            "HC Changes in Title by Department, Region List"]] = \
        df_all.progress_apply(lambda row: department_title_changes(row, emp_group_dept, emp_group_region),
                              axis=1, result_type="expand")
    return df_all


# @task(name="Adding derived performance review features to data", cache_key_fn=task_input_hash)
def compute_derived_review_features(df_all):
    for col in ["T-1", "T-2"]:
        df_all["Reviews - MEDIAN " + col] = df_all[["Manager - MEDIAN " + col,
                                                    'Self - MEDIAN ' + col,
                                                    '360 Reviewer - MEDIAN ' + col]].median(axis=1)

        df_all["Reviews - MEAN " + col] = df_all[["Manager - MEAN " + col,
                                                  'Self - MEAN ' + col,
                                                  '360 Reviewer - MEAN ' + col]].mean(axis=1)

        df_all["Participated In 360 Reviews " + col] = df_all["360 Reviewer - COUNT " + col].apply(
            lambda x: "Did Not Participate" if pd.isna(x) or x == 0 else "Participated")

        df_all["self_exceeds_reviewer_median " + col] = \
            df_all.progress_apply(lambda row: row["Self - MEDIAN " + col] - \
                                              row["360 Reviewer - MEDIAN " + col]
            if not pd.isna(row["Self - MEDIAN " + col]) and \
               not pd.isna(row["360 Reviewer - MEDIAN " + col]) else np.nan, axis=1)
        df_all["self_exceeds_reviewer_mean " + col] = \
            df_all.progress_apply(lambda row: row["Self - MEAN " + col] - \
                                              row["360 Reviewer - MEAN " + col]
            if not pd.isna(row["Self - MEAN " + col]) and \
               not pd.isna(row["360 Reviewer - MEAN " + col]) else np.nan, axis=1)

        df_all["self_exceeds_360_manager_median " + col] = \
            df_all.progress_apply(lambda row: row["Self - MEDIAN " + col] - \
                                              pd.Series([row["360 Reviewer - MEDIAN " + col],
                                                         row["Manager - MEDIAN " + col]]).median(),
                                  axis=1)

        df_all["self_exceeds_360_manager_mean " + col] = \
            df_all.progress_apply(lambda row: row["Self - MEAN " + col] - \
                                              pd.Series([row["360 Reviewer - MEAN " + col],
                                                         row["Manager - MEAN " + col]]).mean(),
                                  axis=1)
        df_all["self_exceeds_manager_median " + col] = \
            df_all.progress_apply(lambda row: row["Self - MEDIAN " + col] - row["Manager - MEDIAN " + col],
                                  axis=1)
        df_all["self_exceeds_manager_mean " + col] = \
            df_all.progress_apply(lambda row: row["Self - MEAN " + col] - row["Manager - MEAN " + col],
                                  axis=1)
    return df_all


def get_changes_ratio_yoy(row):
    latest_changes = []
    for col in row.index:
        historical_ratios = row[col]
        if len(historical_ratios) < 2:
            return [np.nan] * len(row.index)
        latest_changes.append(historical_ratios[-1] - historical_ratios[-2])
    return latest_changes


# @task(name="Adding change in headcount title ratios", cache_key_fn=task_input_hash)
def compute_change_in_title_ratios(df_all):
    # Creating changes in ratios
    df_all[["Analysts/MDs Change YOY T-1",
            "Associates/MDs Change YOY T-1",
            "Analysts/MDs+Dirs Change YOY T-1",
            "Associates/MDs+Dirs Change YOY T-1",
            "Analysts+Associates/MDs+Dirs Change YOY T-1",
            "Analysts/Associates Change YOY T-1",
            "Analysts/Associates+VPs Change YOY T-1"]] = df_all[["Historical HC Analysts/MDs",
                                                                 "Historical HC Associates/MDs",
                                                                 "Historical HC Analysts/MDs+Dirs",
                                                                 "Historical HC Associates/MDs+Dirs",
                                                                 "Historical HC Analysts+Associates/MDs+Dirs",
                                                                 "Historical Analysts/Associates",
                                                                 "Historical Analysts/Associates+VPs"]] \
        .apply(get_changes_ratio_yoy, result_type="expand", axis=1)

    return df_all


def get_average_of_reviews(x, review_type, titles):
    """
    get_average_of_reviews - over managing directors as well as MDs + Directors.
    :param x: pd.DataFrame
    :param review_type: str
    :param titles: list
    """
    x = x[x["Corporate Title"].isin(titles)]
    if len(x[review_type].dropna()) == 0:
        return np.nan
    return x[review_type].dropna().sum() / len(x)


def average_md_reviews(df_all):
    for review_type in ["360 Reviewer - MEAN T-1", "360 Reviewer - MEDIAN T-1"]:
        for title_set in [["Managing Director"],
                          ["Managing Director", "Director"]]:
            rev_region = df_all \
                .groupby(["Year", "Region"], as_index=False) \
                .apply(lambda x: get_average_of_reviews(x, review_type=review_type,
                                                        titles=title_set)) \
                .rename(columns={None: "Average Mean Review"})
            rev_dept = df_all \
                .groupby(["Year", "Region", "Department"], as_index=False) \
                .apply(lambda x: get_average_of_reviews(x, review_type=review_type,
                                                        titles=title_set)) \
                .rename(columns={None: "Average Mean Review"})
            rev_dept[f"Average{review_type.split('Reviewer -')[-1].replace('.1', '')} Review " + ','.join(title_set)] = \
                rev_dept.apply(lambda row: rev_region[(rev_region["Year"] == row["Year"]) & \
                                                      (rev_region["Region"] == row["Region"])][
                    "Average Mean Review"].iloc[0] \
                    if pd.isna(row["Average Mean Review"]) else row["Average Mean Review"], axis=1)
            rev_dept = rev_dept.drop(columns=["Average Mean Review"])
            df_all = df_all.merge(rev_dept, on=["Year", "Region", "Department"], how="left")
    return df_all


def should_comp_var_be_ignored(row, is_t1=True):
    if pd.isna(row["Group"]):
        return 1
    group_split = row["Group"].split()
    if len(group_split) == 2:
        return int(int(group_split[0]) == row["Year"] - (1 if is_t1 else 2))
    else:
        return 0


def remove_hires_from_prev_year(row):
    if pd.isna(row["Hire Date"]):
        return True
    return int(row["Hire Date"].year) != (row["Year"] - 1)


@flow(name="Removing employees that have NA compensation feature(s)")
def clean_up_compensation_variables(df_all, root_path: str, comp_groups_relpath: str):
    # only return False for cases where Hire Date == Year - 1
    df_all = df_all[df_all.apply(remove_hires_from_prev_year, axis=1)]
    comp_groups = pd.read_excel(f"{root_path}/{comp_groups_relpath}")

    df_all = df_all.merge(comp_groups[["ID", "Year", "Group"]],
                          how="left",
                          on=["ID", "Year"])

    # df_all["Should Compensation T-1 Be Ignored?"] = df_all.apply(should_comp_var_be_ignored, axis=1)
    # df_all["Should Compensation T-2 Be Ignored?"] = df_all \
    #     .apply(lambda row: should_comp_var_be_ignored(row, False), axis=1)
    return df_all


######################
# Data Sanity Checks #
######################
class DQA(object):
    """
    Data Quality Assurance (DQA)
    """

    def __init__(self, df):
        self.df = df
        cols_to_drop = []
        for col in self.df:
            if (len(self.df[col].dropna()) == 0 or type(self.df[col].dropna().iloc[0]) == list) and \
                    (col != "Junior ID"):
                cols_to_drop.append(col)
        self.df = self.df.drop(columns=cols_to_drop)
        self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()

    def check_duplicates(self):
        grouped_ids = self.df.groupby(["ID", "Year"], as_index=False) \
            .agg(list)[["ID", "Year", "Date"]]
        duplicate_ids = grouped_ids[grouped_ids.apply(lambda row: len(row["Date"]) > 1, axis=1)]
        assert len(duplicate_ids) == 0, "There exists duplicate employee records by ID, Year in the data."
        print("No duplicates in data!")

    def check_coverage(self, variables, missingness_threshold):
        for variable in variables:
            assert missingness_threshold >= self.df[variable].isnull().sum() / len(self.df[variable]), \
                f"Coverage exceeds missingness threshold: {missingness_threshold}, {variable}"
        print("Coverage is sufficient!")

    def check_no_future_leavers(self):
        '''
        check_no_future_leavers - data quality check that no leavers that have left over the
        course of a given year will still exist in the data, unless they were rehired.

        Store leavers in mapping with termination date. Check if their latest hire date is after
        their termination date for that year.
        '''
        preceding_leavers = set()
        min_year, max_year = self.df["Year"].min(), self.df["Year"].max()
        for year in range(min_year, max_year + 1):
            curr_year_df = self.df[self.df["Year"] == year]
            # Leaver IDs
            curr_ids = set(curr_year_df[curr_year_df["Attrition"] == 1]["ID"])
            overlap_with_past = preceding_leavers.intersection(curr_ids)
            print(f"Overlap for {year}:", overlap_with_past)
            assert len(overlap_with_past) == 0 or \
                   all(curr_year_df[curr_year_df["ID"] == curr_id]["Hire Date"].iloc[0] < datetime(year, 1, 1) \
                       for curr_id in curr_ids), \
                f"Current HC overlapping with past leavers for year {year}: # {overlap_with_past}"
            preceding_leavers.update(curr_ids)
        print("No leavers existing after their termination date!")

    def check_temporal_factors_increasing(self, factors):
        '''
        check_temporal_factors_increasing - check for factors
        that change with time in that they increase with every
        passing year.
        '''
        pass

    def check_nonnegative_numerical(self, factors, minimum=None, maximum=None):
        '''
        check_nonnegative_numerical - check factors list for
        fields that must be >= 0.
        '''
        if minimum is None:
            minimum = 0
        if maximum is None:
            maximum = float("inf")
        for factor in factors:
            assert len(self.df[self.df[factor] < 0]) == 0, f"Negative Numericals for column {factor}"
        print(f"No negative numerical values for the factors: {factors}")

    def check_as_of_after_hire(self):
        assert len(self.df[self.df.apply(lambda row: row["Hire Date"] >= row["Date"], axis=1)]) == 0, \
            "Not all employee hire dates are before the HC date"

    def check_term_after_as_of(self):
        assert len(self.df[self.df.apply(lambda row: (not pd.isna(row["Term Date"])) and \
                                                     (row["Date"] > row["Term Date"]),
                                         axis=1)]) == 0, "Term Date precedes the headcount date of employee"

        termination_hc_year_equality = self.df[self.df.apply(lambda row: (not pd.isna(row["Term Date"])) and \
                                                                         (row["Date"].year != row["Term Date"].year),
                                                             axis=1)]
        assert len(termination_hc_year_equality) == 0, "Termination Date Year Not in Same Year as HC Date:" + \
                                                       f"{termination_hc_year_equality[['Date', 'Term Date']]}"

    def examine_stayer_overlap_duplicates(self, row):
        uniq_items_for_id = row["Date"]
        assert any(row[col] == uniq_items_for_id for col in list(self.df.columns)), \
            "All duplicated rows - repeated information from the past"

    def examine_stayer_overlap(self):
        previous_stayers = dict()
        previous_stayers[2017] = set()
        for year in range(2018, datetime.today().year + 1):
            curr_stayers = self.df[(self.df["Year"] == year) & (self.df["Attrition"] == 0)]["ID"]
            set_stayers = set(curr_stayers)
            overlap = previous_stayers[year - 1].intersection(set_stayers)
            if len(overlap) > 0:
                grouped_overlap = self.df[self.df["ID"].isin(overlap)].copy() \
                    .reset_index(drop=True) \
                    .groupby(["ID"], as_index=False) \
                    .agg(lambda x: len(set(x)))
                for i, row in grouped_overlap.iterrows():
                    self.examine_stayer_overlap_duplicates(row)
            print(f"Overlap of stayers with previous years in {year}: {len(overlap)}," + \
                  f" {len(set_stayers.difference(overlap))}")
            previous_stayers[year] = set_stayers


@flow(name="Running sanity checks over the filtered dataset")
def sanity_checks(df_all, perc_women_column_names, stagnation_features, other_comp_vars=[]):
    personal = ["ID", "Junior ID", "Supv ID", "Sex", "Company", "Age", "Functional Title",
                "Corporate Title",
                "Category",
                "Empl Class", "FTE",
                "Years of Service (bin)", \
                "Department",
                "Bus Unit",
                "Business Area",
                "Age Relative to Title"]
    region = ["Region", "Country", "City"]
    compensation = ["Comp Tier T-1",
                    "YOY TC % Chg T-1",
                    "YOY TC % Chg (bin) T-1"] + other_comp_vars
    culture_clash = ["Turnover Rate by Department, Region",
                     "Turnover Rate by Supervisor",
                     "Turnover in Group Among Juniors",
                     "Turnover in Group Among Seniors"] + perc_women_column_names
    target = ["Attrition"]
    date_var = ["Year", "Hire Date", "Term Date"]
    stagnation_features += ["Average Employees to Manager in Department, Region",
                            "Average Mean Review of Managers in Department, Region"]
    subset_vars = personal + region + stagnation_features + compensation + culture_clash + date_var + target
    fa_revenue_post_2017 = DQA(df_all)

    fa_revenue_post_2017.check_duplicates()
    fa_revenue_post_2017.check_coverage([var for var in subset_vars if \
                                         all(x not in var for x in ["Date", "Comp",
                                                                    "Junior ID",  # variables below are for non-FA
                                                                    "Reviews",
                                                                    "Reviewer",
                                                                    "self_exceeds",
                                                                    "YOY TC",
                                                                    "TC Mkt Data",
                                                                    "Turnover in Group",
                                                                    "Average MEAN T-1 Review",
                                                                    "Average MEDIAN T-1 Review",
                                                                    "Average Employees to Manager",
                                                                    "Average Mean Review of Managers"] + \
                                             perc_women_column_names)],
                                        0.20)
    fa_revenue_post_2017.check_no_future_leavers()
    fa_revenue_post_2017.check_nonnegative_numerical(["Age", "Years In Title"], minimum=0)
    # fa_revenue_post_2017.check_as_of_after_hire()
    fa_revenue_post_2017.check_term_after_as_of()

    fa_revenue_post_2017.examine_stayer_overlap()
    return {"personal": personal,
            "region": region,
            "stagnation": stagnation_features,
            "compensation": compensation,
            "culture_clash": culture_clash,
            "date_var": date_var,
            "target": target}


###############################
# HR Data Ingestion Functions #
###############################
@task(name="Filter the data by Financial Advisory (FA)", cache_key_fn=task_input_hash)
def filter_by_fa(df_all, fa_countries):
    df_all = df_all[(df_all["Bus Unit"] == "Financial Advisory") & \
                    (df_all["Category"] == "Revenue Producing") & \
                    (df_all["Year"] > 2017)]
    # df_all = df_all[df_all["Empl Class"] != 'INT']
    # df_all = df_all[~df_all["Corporate Title"].isin({"Analyst", "Senior Advisor", "Managing Director",
    #                                                  "Not Applicable"})]
    # df_all = df_all[df_all["Functional Title"] != "Associate 0"]

    # df_all = df_all[df_all["Country"].isin(fa_countries)]
    # df_all = df_all[df_all["Hire Date"].apply(lambda x: x.year)
    #                 != df_all["Year"]]
    # df_all = df_all[~df_all["Department"].isin({"Private Capital Advisory",
    #                                             "Investor Relations Advisory",
    #                                             "Venture and Growth Banking",
    #                                             "Data Analytics Group"})]
    # df_all = df_all[df_all["Should Compensation T-1 Be Ignored?"] == 0]
    return df_all


@task(name="Filter by Asset Management (AM)")
def filter_by_am(df_all, fa_countries):
    # we still need the year filter because of review data.

    df_all = df_all[(df_all["Bus Unit"] == "Asset Management") & \
                    (df_all["Category"] == "Revenue Producing") & \
                    (df_all["Year"] > 2017)]
    # excluded intentionally
    df_all = df_all[df_all["Empl Class"] != 'INT']
    df_all = df_all[~df_all["Corporate Title"].isin({"Analyst", "Senior Advisor", "Managing Director",
                                                     "Not Applicable"})]
    df_all = df_all[df_all["Functional Title"] != "Associate 0"]

    df_all = df_all[df_all["Country"].isin(fa_countries)]
    df_all = df_all[df_all["Hire Date"].apply(lambda x: x.year)
                    != df_all["Year"]]
    df_all = df_all[~df_all["Department"].isin({"Private Capital Advisory",
                                                "Investor Relations Advisory",
                                                "Venture and Growth Banking",
                                                "Data Analytics Group"})]
    print("Filtered AM:", len(df_all))
    return df_all


@task(name="Filter by Corporate and other Non Revenue Producing Employees in Other Business Units")
def filter_by_non_revenue(df_all, fa_countries):
    df_all = df_all[(df_all["Category"] == "Non Revenue Producing") & \
                    (df_all["Year"] > 2017)]
    df_all = df_all[df_all["Empl Class"] != 'INT']
    df_all = df_all[~df_all["Corporate Title"].isin({"Analyst", "Senior Advisor", "Managing Director",
                                                     "Not Applicable"})]
    df_all = df_all[df_all["Functional Title"] != "Associate 0"]

    df_all = df_all[df_all["Country"].isin(fa_countries)]
    df_all = df_all[df_all["Hire Date"].apply(lambda x: x.year)
                    != df_all["Year"]]
    df_all = df_all[~df_all["Department"].isin({"Private Capital Advisory",
                                                "Investor Relations Advisory",
                                                "Venture and Growth Banking",
                                                "Data Analytics Group"})]
    return df_all


@flow(name="Load General Data Files")
def load_data(fpath, department_map, region_map):
    df_leavers = pd.read_excel(fpath, "Leavers")
    df_leavers = parse_hr_data(df_leavers, year_col='Term Date',
                               department_map=department_map,
                               region_map=region_map)

    # Load all headcounts and get YE employees.
    # Note: df_all contains the leavers.

    df_all1 = pd.read_excel(fpath, "HC")
    df_all2 = pd.read_excel(fpath, "HC2")
    df_all3 = pd.read_excel(fpath, "HC3")

    df_all = pd.concat([df_all1, df_all2, df_all3])
    df_all = parse_hr_data(df_all, year_col='Date',
                           department_map=department_map,
                           region_map=region_map)

    starts = df_all[df_all["Date"].isin(df_all.groupby("Year").Date.min())]
    starts = starts[starts["Date"].apply(lambda x: x.day == 1 and x.month == 1)]

    ends = df_all[df_all["Date"].isin(df_all.groupby("Year").Date.max())]
    ends = ends[ends["Date"].apply(lambda x: x.day == 31 and x.month == 12)]
    ends["Year"] = ends["Year"] + 1

    leaver_columns = ["ID", "Year", "Reason", "Reason.1", "TermType", "Term Date"]

    df_all = starts.loc[:, ~starts.columns.isin({"Term Date", "Resignation",
                                                 "Action",
                                                 "Last Date Worked",
                                                 "Reason",
                                                 "Reason.1"})]
    # df_all = ends.loc[:, ~ends.columns.isin({"Term Date", "Resignation",
    #                                              "Action",
    #                                              "Last Date Worked",
    #                                              "Reason",
    #                                              "Reason.1"})]

    df_all = df_all.merge(df_leavers[leaver_columns],
                          how="left",
                          on=["ID", "Year"])
    df_all = df_all.sort_values("Term Date").drop_duplicates(["ID", "Year"], keep="last")
    df_all["Attrition"] = df_all["TermType"].apply(lambda x: int(x == "VOLUNT"))
    return df_all


@flow(name="Load Additional Files")
def load_additional_data(base_data, fpaths, department_map, region_map, all_cols):
    hc_lst = []
    leaver_lst = []
    leaver_columns = ["ID", "Year", "Reason", "Reason.1", "TermType", "Term Date"]
    for fpath in fpaths:
        df_leavers = pd.read_excel(fpath, "Leavers")
        df_leavers = parse_hr_data(df_leavers, year_col='Term Date',
                                   department_map=department_map,
                                   region_map=region_map)
        # Load all headcounts and get YE employees.
        # Note: df_all contains the leavers.
        df_all = pd.read_excel(fpath, "HC")
        df_all = parse_hr_data(df_all, year_col='Date',
                               department_map=department_map,
                               region_map=region_map)

        starts = df_all[df_all["Date"].isin(df_all.groupby("Year").Date.min())]
        starts = starts[starts["Date"].apply(lambda x: x.day == 1 and x.month == 1)]

        ends = df_all[df_all["Date"].isin(df_all.groupby("Year").Date.max())]
        ends = ends[ends["Date"].apply(lambda x: x.day == 31 and x.month == 12)]
        ends["Year"] = ends["Year"] + 1

        df_all = starts.loc[:, ~starts.columns.isin({"Term Date", "Resignation",
                                                     "Action",
                                                     "Last Date Worked",
                                                     "Reason",
                                                     "Reason.1"})]
        # df_all = ends.loc[:, ~ends.columns.isin({"Term Date", "Resignation",
        #                                              "Action",
        #                                              "Last Date Worked",
        #                                              "Reason",
        #                                              "Reason.1"})]

        hc_lst.append(df_all)
        leaver_lst.append(df_leavers[leaver_columns])
    all_leavers = pd.concat(leaver_lst)
    all_hc = pd.concat(hc_lst)
    na_cols = set(base_data.columns).difference(set(all_hc.columns))
    all_hc[list(na_cols)] = np.nan
    # Merge the new headcounts with the base data.
    new_data = pd.concat([base_data, all_hc[base_data.columns]])
    new_data = new_data.merge(all_leavers, how="left", on=["ID", "Year"])

    for col in leaver_columns:
        if col not in ["ID", "Year"]:
            new_data[col] = new_data[f"{col}_y"].fillna(new_data[f"{col}_x"])

    new_data = new_data.sort_values("Term Date").drop_duplicates(["ID", "Year"], keep="last")
    new_data["Attrition"] = new_data["TermType"].apply(lambda x: int(x == "VOLUNT"))
    new_data = new_data[all_cols]
    return new_data


@task(name="Adding T-1, T-2 historical year columns to Performance Reviews Data")
def add_historical_years(i, min_year, processed_df, review_vars):
    processed_df["T-1 Year"] = min_year + i + 1
    processed_df["T-2 Year"] = min_year + i + 2
    processed_df = processed_df.rename(columns={"Reviewee Employee ID": "ID"})
    processed_df = processed_df[review_vars]
    return processed_df


def read_review_excel(root_path, performance_review_relpath, year):
    try:
        return pd.read_excel(f"{root_path}/{performance_review_relpath}", str(year), skiprows=1)
    except Exception as e:
        print(e)
        return None


@flow(name="Load Performance Reviews")
def load_reviews(df_all, root_path, performance_review_relpath, review_vars):
    """
    load_review
    :param df_all:
    :param root_path:
    :param performance_review_relpath:
    :param review_vars:
    :return:
    """
    min_year, max_year = df_all["Year"].min(), df_all["Year"].max()
    processed_reviews = [(year, read_review_excel(root_path, performance_review_relpath, year))
                         for year in range(min_year, max_year + 1)]
    processed_reviews = [(year, review_df) for (year, review_df) in processed_reviews if review_df is not None]
    min_year = processed_reviews[0][0]
    processed_reviews = [review_df for _, review_df in processed_reviews]
    processed_reviews = [processed_df.rename(columns={"Unnamed: 0": processed_df.iloc[0]['Unnamed: 0']}).iloc[1:]
                         for i, processed_df in enumerate(processed_reviews)]
    processed_reviews = [add_historical_years(i, min_year, processed_df, review_vars)
                         for i, processed_df in enumerate(processed_reviews)]
    processed_reviews = pd.concat(processed_reviews)

    get_shift_vars = [var for var in review_vars if var not in {"ID", "T-1 Year",
                                                                "T-2 Year"}]

    for i, col in enumerate(["T-1 Year", "T-2 Year"]):
        get_shift_vars_i = [var + f" T-{i + 1}" for var in get_shift_vars]
        rename_dict = dict(zip(get_shift_vars, get_shift_vars_i))
        rename_dict[col] = "Year"
        df_all = df_all.merge(processed_reviews[["ID", col] + get_shift_vars].rename(columns=rename_dict),
                              how="left",
                              on=["ID", "Year"])
    return df_all


def modify_yoy_tc_chg(chg):
    if type(chg) == float and chg == float('-inf'):
        return "None"
    if type(chg) == str or pd.isna(chg):
        return chg
    assert type(chg) in {float, int}, "Change not numeric"
    for val in range(-4, 5, 1):
        if chg < val / 4:
            if val == -4:
                return "-100%+"
            else:
                return str((val - 1) * 100 / 4) + "% - " + str(val * 100 / 4) + "%"
    return "100%+"


@flow(name="Load Compensation")
def load_compensation(df_all, root_path: str, compensation_data_relpath: str, additional_relpaths):
    df_comps = pd.read_excel(f"{root_path}/{compensation_data_relpath}", "Population")
    addon_comps = []
    for path in additional_relpaths:
        addon_comps.append(pd.read_excel(f"{root_path}/{path}"))
    addon_comps = pd.concat(addon_comps)
    df_comps = pd.concat([df_comps, addon_comps[df_comps.columns]])

    df_comps["T-1 Year"] = df_comps["Year"] + 1
    df_comps["T-2 Year"] = df_comps["Year"] + 2
    get_shift_vars = [var for var in df_comps.columns if var not in {"Empl ID", "Year", "T-1 Year", "T-2 Year"}]
    print("# Concatenated Compensation Dataframe #")
    print(df_comps)
    for j, col in enumerate(["T-1 Year", "T-2 Year"]):
        get_shift_vars_i = [var + f" T-{j + 1}" for var in get_shift_vars]
        rename_dict = dict(zip(get_shift_vars, get_shift_vars_i))
        rename_dict[col] = "Year"
        rename_dict["Empl ID"] = "ID"
        df_all = df_all.merge(df_comps[["Empl ID", col] + get_shift_vars].rename(columns=rename_dict),
                              how="left",
                              on=["ID", "Year"])

    return df_all


@flow(name="Load Promotions")
def load_promotions(df_all, root_path: str, promotions_data_relpath: str):
    df_promotions = pd.read_excel(f"{root_path}/{promotions_data_relpath}", "Promotions")

    # Amount of time since employee's last promotion.

    unique_df_promotions = df_promotions.sort_values("Eff Date", ascending=False)
    promotion_data = pd.DataFrame()
    promotion_data["ID"] = df_all["ID"].values
    promotion_data["Last Promotion Date"] = get_last_promotion_date_df(df_all, unique_df_promotions)
    df_all = df_all.fillna(promotion_data)
    return df_all


@flow(name="Prepare Flight Risk Data")
def prepare_flight_risk_data(root_path: str, relpath_dict: dict, just_read_data: bool = True):
    """
    prepare_flight_risk_data only takes in file paths that need to be loaded,
    which contain general hr factors, compensation, and performance reviews.
    Artifacts produced from the function include a missingness report
    and a short excel to be appended to a master spreadsheet.

    :param root_path: str
    :param relpath_dict: str
    :return: pd.DataFrame
    """
    assert len(set(relpath_dict.keys()).difference({"dept_relpath",
                                                    "region_relpath",
                                                    "fa_countries_relpath",
                                                    "review_vars_relpath",
                                                    "general_data_relpath",
                                                    "performance_review_relpath",
                                                    "compensation_data_relpath",
                                                    "additional_comp_relpaths",
                                                    "promotions_data_relpath",
                                                    "pre2020_relpath",
                                                    "ye2021_relpath",
                                                    "comp_groups_relpath",
                                                    "subset_stagnation_features_relpath",
                                                    "all_stagnation_features_relpath",
                                                    "additional_data_relpaths"})) == 0, ""

    review_cols = json.load(open(f"{root_path}/{relpath_dict['review_vars_relpath']}", "r"))
    fa_countries = json.load(open(f"{root_path}/{relpath_dict['fa_countries_relpath']}", "r"))
    department_map = json.load(open(f"{root_path}/{relpath_dict['dept_relpath']}", "r"))
    region_map = json.load(open(f"{root_path}/{relpath_dict['region_relpath']}", "r"))
    subset_stagnation_features = json.load(open(f"{root_path}/{relpath_dict['subset_stagnation_features_relpath']}",
                                                "r"))
    all_stagnation_features = json.load(open(f"{root_path}/{relpath_dict['all_stagnation_features_relpath']}", "r"))

    general_data_path = f"{root_path}/{relpath_dict['general_data_relpath']}"
    df_all = load_data(general_data_path, department_map, region_map)

    df_all["Last Promotion Date"] = np.nan
    df_all = df_all.drop(columns=['Home Zipcode', 'Office Zipcode', 'Zipcode'])
    df_all = load_additional_data(df_all, relpath_dict["additional_data_relpaths"],
                                  department_map,
                                  region_map,
                                  df_all.columns)

    if just_read_data:
        df_all.to_excel(f"{root_path}/data/ref/all_employees_base.xlsx", "Employees", index=False)
        print("Wrote to all employees - includes Attrition status of all leavers/stayers.")
        return df_all, None, None
    df_all = load_reviews(df_all, root_path, relpath_dict['performance_review_relpath'], review_cols)

    df_all = load_compensation(df_all, root_path, relpath_dict['compensation_data_relpath'],
                               relpath_dict["additional_comp_relpaths"])

    # Adding Features #
    df_all = load_promotions(df_all, root_path, relpath_dict["promotions_data_relpath"])
    df_all["Years In Title"] = df_all.apply(get_yrs_in_title, axis=1)
    df_all["Age"] = df_all.apply(map_age, axis=1)
    df_all = compensation_perc_change(df_all)
    df_all, emp_supv_groups = group_attrition(df_all)
    df_all = manager_specific(df_all, root_path, relpath_dict["pre2020_relpath"], relpath_dict["ye2021_relpath"])
    df_all = compensation_related_features(df_all)
    df_all = headcount_changes(df_all)
    df_all, emp_group_dept, emp_group_region = headcount_ratios(df_all)

    df_all = relative_attrition(df_all, emp_group_dept, emp_group_region)

    df_all = compute_relative_age_to_title(df_all, emp_group_dept, emp_group_region)
    df_all, perc_women_column_names = compute_perc_women_factors(df_all, emp_supv_groups,
                                                                 emp_group_dept, emp_group_region)

    df_all = compute_manager_specific(df_all, emp_group_dept, emp_group_region)

    df_all = title_related_features(df_all, emp_group_dept, emp_group_region)
    df_all = compute_derived_review_features(df_all)

    df_all = compute_change_in_title_ratios(df_all)
    df_all = average_md_reviews(df_all)

    # Filtering and Cleaning up the Rows
    df_all = clean_up_compensation_variables(df_all, root_path, relpath_dict["comp_groups_relpath"])
    df_filtered = filter_by_fa(df_all, fa_countries)
    # df_filtered = filter_by_am(df_all, fa_countries)
    # df_filtered = filter_by_non_revenue(df_all, fa_countries)
    print(f"{len(df_filtered)} Employee Records")
    # Sanity checks
    subset_vars_dict = sanity_checks(df_filtered, perc_women_column_names, subset_stagnation_features,
                                     other_comp_vars=["TC Mkt Data Quartile T-1", "TC Mkt Data Quartile Change T-1"])
    var_categories, all_vars = export_dataframe(root_path, df_filtered, df_all, all_stagnation_features,
                                                perc_women_column_names, subset_vars_dict, write_to_excel=True,
                                                other_comp_vars=["TC Mkt Data Quartile T-1",
                                                                 "TC Mkt Data Quartile Change T-1"])
    missingngess_excel_relpath = 'output/flight_risk_predictors_na_summary-fa-revenue.xlsx'
    curr_years, all_historicals_data_na, \
        some_historicals_data_na = generate_missingness_excel(df_filtered, all_vars, var_categories, root_path,
                                                              output_relpath=missingngess_excel_relpath,
                                                              write_to_excel=True)
    fig_list = generate_missingness_report(curr_years, all_historicals_data_na, some_historicals_data_na,
                                           var_categories)
    _ = _write_plotly_go_to_html(fig_list)
    return df_all, df_filtered, subset_vars_dict


def _write_plotly_go_to_html(fig_list, output_filepath="output/flight_risk_predictors_summary.html"):
    """
    Write list of plotly graph objects to html
    """
    with open(output_filepath, "w") as f:
        for fig in fig_list:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        html = codecs.open(output_filepath, "r", "utf-8").read()

    return html


@flow(name="Exporting dataframe to csv for visuals")
def export_for_visuals(df_all, subset_vars_dict, root_path, output_relpath):
    # for visualizations
    subset_vars = list(functools.reduce(lambda x, y: x + y, subset_vars_dict.values()))
    df_export = df_all.copy()
    df_export[["Comp Tier T-2"] + subset_vars + ['self_exceeds_reviewer_median T-1',
                                                 'self_exceeds_reviewer_mean T-1',
                                                 "self_exceeds_360_manager_median T-1",
                                                 "self_exceeds_360_manager_mean T-1",
                                                 "self_exceeds_manager_median T-1",
                                                 "self_exceeds_manager_mean T-1"]].to_csv(
        f"{root_path}/{output_relpath}")


@flow(name="Exporting dataframe to Excel")
def export_dataframe(root_path, df_filtered, df_all, all_stagnation_features, perc_women_column_names, subset_vars_dict,
                     write_to_excel=True, other_comp_vars=[]):
    # Company can be a proxy for location (Country/City)
    personal_vars = ["ID", "Sex", "Company", "Age", "Functional Title", "Corporate Title",
                     "Category", "Empl Class", "FTE", "Years of Service (bin)",
                     "Bus Unit", "Department",
                     "Business Area", "Supv ID",
                     "Group",
                     "Age Relative to Title"]

    location_vars = ["Region", "Country", "City"]
    date_vars = ["Term Date", "Hire Date", "Year"]

    comp_vars = ["YOY TC % Chg T-1", "YOY TC % Chg T-2",
                 "Comp Tier T-1", "Comp Tier T-2",
                 "YOY TC % Chg (bin) T-1",
                 "YOY TC % Chg (bin) T-2",
                 "YOY TC % Chg CAGR",
                 "Comp Tier Change T-1",
    "Should Compensation T-1 Be Ignored?",
    "Should Compensation T-2 Be Ignored?"]
    comp_vars += other_comp_vars

    culture_clash_vars = ["Turnover Rate by Department, Region",
                          "Turnover Rate by Supervisor",
                          "Turnover in Group Among Juniors",
                          "Turnover in Group Among Seniors"] + perc_women_column_names

    all_stagnation_features += ["Average Employees to Manager in Department, Region",
                                "Average Mean Review of Managers in Department, Region"]

    work_life_imbalance = []
    professional_profile = []

    target_var = "Attrition"
    var_categories = ["Personal Attributes"] * len(personal_vars) + \
                     ["Role Stagnation"] * len(all_stagnation_features) + \
                     ["Location"] * len(location_vars) + \
                     ["Compensation"] * len(comp_vars) + \
                     ["Culture Clash/Work Life Imbalance"] * ((len(culture_clash_vars))) + \
                     ["Professional Profile"] * len(professional_profile) + \
                     ["Date"] * len(date_vars) + ["Target"]
    all_vars = personal_vars + all_stagnation_features + location_vars + comp_vars + culture_clash_vars + \
               work_life_imbalance + professional_profile + date_vars + [target_var]

    if not write_to_excel:
        return var_categories, all_vars
    df_all["YOY TC % Chg (bin) T-1"] = df_all["YOY TC % Chg (bin) T-1"].replace("None", np.nan)

    subset_vars = list(functools.reduce(lambda x, y: x + y, subset_vars_dict.values()))

    temp = df_filtered[subset_vars]
    # some NaTs are actually NA.
    temp = temp.replace("NaT", np.nan)

    excluded_temp = df_all[["ID", "Year"] + list(set(all_vars).difference(set(subset_vars))) + ["Hire Date",
                                                                                                "Attrition",
                                                                                                "Term Date"]]
    excluded_temp = excluded_temp.replace("NaT", np.nan)

    excluded_temp_columns = []
    t1_cols, t2_cols = [], []
    for col in list(excluded_temp.columns):
        if "T-1" in col or "T-2" in col:
            if "T-1" in col:
                assert "T-2" not in col, "Both T-1, T-2 are in the column name!"
                t1_cols.append(col)
            else:
                t2_cols.append(col)
        else:
            excluded_temp_columns.append(col)
    excluded_temp_columns.extend(t1_cols)
    excluded_temp_columns.extend(t2_cols)

    writer = pd.ExcelWriter(f'{root_path}/output/HR Flight Risk Predictors.xlsx', engine='xlsxwriter')
    temp.to_excel(writer, sheet_name="Flight Risk Predictors Included", startrow=0, index=False)
    excluded_temp[excluded_temp_columns].to_excel(writer,
                                                  sheet_name="Appendix - Predictors Excluded",
                                                  startrow=0,
                                                  index=False)
    df_all.to_excel(writer, sheet_name="Appendix - All Data",
                    startrow=0,
                    index=False)
    writer.save()
    return var_categories, all_vars


@flow(name="generate missingness excel report for dataframe")
def generate_missingness_excel(df_all, all_vars, var_categories, root_path,
                               output_relpath='flight_risk_predictors_na_summary.xlsx',
                               write_to_excel=True):
    # Saving to workbook
    if write_to_excel:
        writer = pd.ExcelWriter(f"{root_path}/{output_relpath}", engine='xlsxwriter')
    # Saving NA distribution and variable coverage to Excel/Plotly

    curr_years = []
    all_historicals_data_na = []
    some_historicals_data_na = []
    marker_color_to_year = \
        {2023: "rgb(145,145,145)",
         2022: "rgb(190,190,190)",
         2021: "rgb(205,220,230)",
         2020: "rgb(95,135,160)",
         2019: "rgb(207,160,82)",
         2018: "rgb(122,167,120)",
         2017: "rgb(134,186,132)",
         2016: "rgb(145,201,143)",
         2015: "rgb(158,219,156)",
         2014: "rgb(176,245,174)"}
    # Checking for overlaps between individual years of stayers/leavers.
    min_year = df_all["Year"].min()
    max_year = df_all["Year"].max()
    for year in range(min_year, max_year + 1):
        curr_year = df_all[df_all["Year"] == year].reset_index()
        curr_year = curr_year[(curr_year["Hire Date"].apply(lambda x: x.year if not pd.isna(x) else x) < year)]
        curr_year = curr_year.rename(columns={"Action": "Job Transfer Action",
                                              "time_in_seconds": "Commute Duration (Seconds)",
                                              "dist_in_meters": "Commute Distance (Meters)"})
        curr_year = curr_year[all_vars]
        uniq_year = curr_year.drop_duplicates(["ID"]).reset_index()

        curr_years.append(curr_year)
        curr_stayers = uniq_year[uniq_year["Attrition"] == 0]
        no_na_comp_tier_stayers = curr_stayers[~pd.isna(curr_stayers["Comp Tier T-1"])]

        curr_leavers = uniq_year[uniq_year["Attrition"] == 1]
        print(f"{year} - curr_leavers_this_year columns", curr_leavers.columns)
        curr_leavers_this_year = curr_leavers[curr_leavers["Term Date"].apply(lambda x: x.year == year)]
        if len(curr_leavers_this_year) > 0:
            leavers_this_year_ids = set(curr_leavers_this_year["ID"])
            curr_leavers_not_this_year = curr_leavers[~curr_leavers["ID"].isin(leavers_this_year_ids)]
            no_na_comp_tier_leavers = curr_leavers[~pd.isna(curr_leavers["Comp Tier T-1"])]
        else:
            curr_leavers_not_this_year = pd.DataFrame()
            no_na_comp_tier_leavers = pd.DataFrame()
        if year == min_year:
            stayers_overlapping_prev = 0
            leavers_overlapping_prev = 0
        else:
            stayers_overlapping_prev = len(set(curr_stayers["ID"]).intersection(set(prev_stayers["ID"])))
            leavers_overlapping_prev = len(set(curr_leavers["ID"]).intersection(set(prev_leavers["ID"])))
        print(f"#######\nYear: {year}\nNumber of Unique Employees: {len(uniq_year)}" + \
              f"\nNumber of Unique Stayers: {len(curr_stayers)}" + \
              f"\nNumber of Unique Leavers Who Left This Year: {len(curr_leavers_this_year)}" + \
              f"\nNumber of Unique Leavers Who Will Leave, But Not This Year: {len(curr_leavers_not_this_year)}" + \
              f"\nNumber of Available Comp Tiers (Unique) in Leavers: {len(no_na_comp_tier_leavers)}" + \
              f"\nNumber of Available Comp Tiers (Unique) in Stayers: {len(no_na_comp_tier_stayers)}" + \
              f"\nNumber of Stayers Overlapping With Previous Year: {stayers_overlapping_prev}" + \
              f"\nNumber of Leavers Overlapping With Previous Year: {leavers_overlapping_prev}")

        # Distribution of NA values
        na_values_some = dict()
        for col in curr_year:
            shrunk_column = curr_year.drop_duplicates(["ID", "Year"])
            na_values_some[col] = [np.round(shrunk_column[col].isnull().sum() * 100 / len(shrunk_column), 2)]
        na_values_some = pd.DataFrame(na_values_some)
        # Assumed that there are no duplicated columns
        grouped_curr_year = curr_year.groupby("ID")
        na_values_all = pd.DataFrame(grouped_curr_year.agg(lambda x: all(pd.isna(x))).sum(axis=0))

        na_values_all = na_values_all.rename(columns={0: "NA values as % of Total"})

        na_values_some = na_values_some.drop(columns=["ID"])
        na_values_some = na_values_some.T.rename(columns={0: "NA values as % of Total"})

        na_values_all = np.round(na_values_all * 100 / len(grouped_curr_year), 2)

        # Truncate the first element off the var_categories as we exclude missingness
        # over IDs (all will be available).
        all_historicals_data_na.append(go.Bar(name=f"{year}", y=[np.array(var_categories[1:]),
                                                                 np.array(list(na_values_all.index))],
                                              x=na_values_all["NA values as % of Total"].values,
                                              orientation="h",
                                              marker_color=marker_color_to_year[year],
                                              marker_line_color='rgb(8,48,107)',
                                              marker_line_width=1.5, opacity=0.6))

        some_historicals_data_na.append(go.Bar(name=f"{year}", y=[np.array(var_categories[1:]),
                                                                  np.array(list(na_values_some.index))],
                                               x=na_values_some["NA values as % of Total"].values,
                                               orientation="h",
                                               marker_color=marker_color_to_year[year],
                                               marker_line_color='rgb(8,48,107)',
                                               marker_line_width=1.5, opacity=0.6))
        if write_to_excel:
            na_values_all.T.to_excel(writer, sheet_name=f"{year}")
            na_values_some.T.to_excel(writer, sheet_name=f"{year}", startrow=4)
            curr_year.to_excel(writer, sheet_name=f"{year}", startrow=6, index=False)

        prev_stayers = curr_stayers
        prev_leavers = curr_leavers
    if write_to_excel:
        writer.save()
    return curr_years, all_historicals_data_na, some_historicals_data_na


@task(name="generate missingness report", cache_key_fn=task_input_hash)
def generate_missingness_report(curr_years, all_historicals_data_na, some_historicals_data_na, var_categories):
    uniq_years_df = pd.concat(curr_years)
    na_values_all = pd.DataFrame(uniq_years_df.groupby("ID").agg(lambda x: all(pd.isna(x))).sum(axis=0))
    na_values_all = na_values_all.rename(columns={0: "NA values as % of Total"})
    na_values_all = np.round(na_values_all * 100 / len(uniq_years_df.groupby("ID")), 2)

    all_historicals_data_na.append(
        go.Bar(name="All Years", y=[np.array(var_categories[1:]), np.array(list(na_values_all.index))],
               x=na_values_all["NA values as % of Total"].values,
               orientation="h",
               marker_color="rgb(128,100,162)",
               marker_line_color='rgb(8,48,107)',
               marker_line_width=1.5, opacity=0.6))
    na_values_some = dict()
    for col in uniq_years_df:
        shrunk_column = uniq_years_df.drop_duplicates(["ID", "Year"])
        na_values_some[col] = [np.round(shrunk_column[col].isnull().sum() * 100 / len(shrunk_column), 2)]
    na_values_some = pd.DataFrame(na_values_some)
    na_values_some = na_values_some.drop(columns=["ID"])
    na_values_some = na_values_some.T.rename(columns={0: "NA values as % of Total"})

    some_historicals_data_na.append(go.Bar(name=f"All Years", y=[np.array(var_categories[1:]),
                                                                 np.array(list(na_values_some.index))],
                                           x=na_values_some["NA values as % of Total"].values,
                                           orientation="h",
                                           marker_color="rgb(128,100,162)", marker_line_color='rgb(8,48,107)',
                                           marker_line_width=1.5, opacity=0.6))
    fig_list = []
    fig = go.Figure(data=some_historicals_data_na)
    uniq_id_temp = uniq_years_df.drop_duplicates("ID")
    fig.update_layout(title=f"<b>Missingness of flight risk predictors as % of Total Employees" + \
                            f" [Leavers: {len(uniq_id_temp[uniq_id_temp['Attrition'] == 1])}," + \
                            f" Stayers: {len(uniq_id_temp[uniq_id_temp['Attrition'] == 0])}]" + \
                            "<br>*Only US, Canada and Europe, Excluding Analysts, Senior Advisors, Managing Directors, Associate 0's," +
                            "<br>Employees Hired in the Same Year as Headcount, and employees from groups:<br>" +
                            "Private Capital Advisory, Investor Relations Advisory, Venture and Growth Banking" + \
                            ", and Data Analytics Group." + "</b>",
                      margin=dict(t=200),
                      yaxis_title="<b>Flight Predictors</b>",
                      xaxis_title=f"<b>NA as % of Total Leavers/Stayers</b>",
                      yaxis_tickformat='.0%',
                      width=1350,
                      height=4000,
                      xaxis_range=[0, 100],
                      font=dict(size=10))
    fig_list.append(fig)
    return fig_list


if __name__ == "__main__":
    relpath_dict = {"dept_relpath": "data/ref/departments.json",
                    "region_relpath": "data/ref/regions.json",
                    "fa_countries_relpath": "data/ref/fa_countries.json",
                    "review_vars_relpath": "data/ref/review_vars.json",
                    "general_data_relpath": "data/ref/LDAG_Files_V5.xls",
                    # TODO: add updated performance review relpath to a list (for 2023 employees).
                    "performance_review_relpath": "data/ref/Flight Risk Performance Summary v11R 2022-04-12.xlsx",
                    "compensation_data_relpath": "data/ref/LDAG Flight Risk Data.xlsx",
                    "additional_comp_relpaths": ["data/ref/LDAG_Comp_2023.xlsx"],
                    "promotions_data_relpath": "data/ref/LDAG_Files_V5.xls",
                    "pre2020_relpath": "data/ref/Sentiment Analysis HR v10R 2022-04-12.csv",
                    "ye2021_relpath": "data/ref/Sentiment Analysis HR 2021YE Reviews v20R 2021-12-23.csv",
                    "comp_groups_relpath": "data/ref/all_fa_employee_data_with GROUP.xlsx",
                    "subset_stagnation_features_relpath": "data/ref/subset_stagnation_features.json",
                    "all_stagnation_features_relpath": "data/ref/all_stagnation_features.json",
                    "additional_data_relpaths": ["data/ref/LDAG_Files_V6.xlsx",
                                                 "data/ref/LDAG_2023.xlsx"]}

    root_path = extract_root_fpath(root_name='llama')
    df_all, df_filtered, subset_vars_dict = prepare_flight_risk_data(root_path=root_path,
                                                                     relpath_dict=relpath_dict,
                                                                     just_read_data=True)
    if df_filtered is not None and subset_vars_dict is not None:
        _ = export_for_visuals(df_filtered, subset_vars_dict, root_path,
                               "data/raw/df_all_perf_comps-fa.csv")
