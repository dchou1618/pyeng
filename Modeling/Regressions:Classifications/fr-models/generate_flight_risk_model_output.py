import codecs
import collections

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import wordninja
from plotly.subplots import make_subplots
from sklearn import metrics

import Lazard_looknfeel_v2 as laz
from preprocess_flight_risk_data import extract_root_fpath

laz_colours = []
for colour in ['cephalopod ink', 'moonstone', 'moonstone_40', 'gold', 'gold_40']:
    laz_colours += [laz.primary_palette[colour]]
for colour in ['hunter', 'moss', 'olive', 'ocean', 'bark', 'tangerine', 'daffodil', 'lilac', 'chestnut']:
    laz_colours += [laz.extended_palette[colour]]

FA_COLUMN_ORDER = ["Employee ID", "Age Relative to Title (Z-score)",
                   "Latest Compensation Tier",
                   "Previous Year's Turnover Rate by Supervisor",
                   "Years In Title",
                   "Employee Headcount by Business Area and Region",
                   "Latest YOY Total Compensation Percentage Change",
                   "Employee Headcount by Supervisor",
                   "Corporate Title",
                   "Latest Median 360 Debiased Review",
                   "Latest Difference Between Debiased Self and Median Reviews (360, Manager)",
                   "Latest Debiased 360 Review Variance",
                   "Latest Percentage Change in Women by Department",
                   "Associates/MDs Organization Structure (Ratio)",
                   "Latest YOY Associate MDs Change (Difference in Ratios)",
                   "Average Employees that a Manager Oversees By Department"]
FA_ALL_RENAME_DICT = {"Age.Relative.to.Title": "Age Relative to Title (Z-score)",
                      "ID": "Employee ID",
                      "Turnover.Rate.by.Supervisor": "Previous Year's Turnover Rate by Supervisor",
                      "Years.In.Title": "Years In Title",
                      "HC.by.Business.Area..Region": "Employee Headcount by Business Area and Region",
                      "YOY.TC...Chg.T.1": "Latest YOY Total Compensation Percentage Change",
                      "HC.by.Supervisor": "Employee Headcount by Supervisor",
                      "Corporate.Title": "Corporate Title",
                      "X360.Reviewer...MEDIAN.T.1": "Latest Median 360 Debiased Review",
                      "self_exceeds_360_manager_median.T.1": \
                          "Latest Difference Between Debiased Self and Median Reviews (360, Manager)",
                      "X360.Reviewer...STD.T.1": "Latest Debiased 360 Review Variance",
                      "latest_perc_change_women_by_dept": "Latest Percentage Change in Women by Department",
                      "Associates.MDs.Organization.Structure": "Associates/MDs Organization Structure (Ratio)",
                      "Associates.MDs.Change.YOY.T.1": "Latest YOY Associate MDs Change (Difference in Ratios)",
                      "Average.Employees.to.Manager.in.Department..Region": \
                          "Average Employees that a Manager Oversees By Department",
                      "Comp.Tier.T.1": "Latest Compensation Tier"}

FA_REVERSE_RENAME_DICT = {FA_ALL_RENAME_DICT[key]: key for key in FA_ALL_RENAME_DICT}

FA_CATEGORICALS = {"Latest Compensation Tier": "<=2",
                   "Associates/MDs Organization Structure (Ratio)": "Bottom Heavy",
                   "Corporate Title": "Associate"}


def get_attritioned_groups(attrition_class):
    prop_attrition = len([cls for cls in attrition_class if cls == 1])
    return prop_attrition, len(attrition_class) - prop_attrition


def get_precision_recall_f1(mat):
    precision = (mat[1, 1]) / (mat[1, 1] + mat[1, 0])
    recall = (mat[1, 1]) / (mat[1, 1] + mat[0, 1])
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def get_auc(true_attrition, prob_attrition):
    fpr, tpr, _ = metrics.roc_curve(true_attrition,
                                    prob_attrition,
                                    pos_label=1)
    return metrics.auc(fpr, tpr)


def get_metrics(mat, true_attrition, prob_attrition):
    precision, recall, f1_score = get_precision_recall_f1(mat)
    auc = get_auc(true_attrition, prob_attrition)
    return precision, recall, f1_score, auc


def map_predictions(lst):
    new_lst = []
    for val in lst:
        if val < 0.25:
            category = "0-25%"
        elif val < 0.5:
            category = "25-50%"
        elif val < 0.75:
            category = "50-75%"
        else:
            category = "75-100%"
        new_lst.append(category)
    return new_lst


def generate_metrics_across_years(mats, prob_dict, test_data_dict, years):
    precisions, recalls, f1_scores, aucs, years_lst = [], [], [], [], []
    for i, year in enumerate(years):
        # how many predictions were made - excluding 2020, we have 3 historical years: 2019, 2021, 2022.
        mat = mats[0][0][i]
        prob_attrition = prob_dict[year]
        true_attrition = test_data_dict[year]["Attrition"]

        precision, recall, f1_score, auc = get_metrics(mat, true_attrition, prob_attrition)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        aucs.append(auc)
        years_lst.append(year)
    return pd.DataFrame({"Precision": precisions, "Recall": recalls,
                         "F1 score": f1_scores, "AUC": aucs,
                         "Year": years_lst}).replace(np.nan, 0).sort_values("Year")


def get_precision_recall_f1(mat):
    precision = (mat[1, 1]) / (mat[1, 1] + mat[1, 0])
    recall = (mat[1, 1]) / (mat[1, 1] + mat[0, 1])
    f1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1_score


def get_auc(true_attrition, prob_attrition):
    fpr, tpr, _ = metrics.roc_curve(true_attrition,
                                    prob_attrition,
                                    pos_label=1)
    return metrics.auc(fpr, tpr)


def get_metrics(mat, true_attrition, prob_attrition):
    precision, recall, f1_score = get_precision_recall_f1(mat)
    auc = get_auc(true_attrition, prob_attrition)
    return precision, recall, f1_score, auc


def read_mat_files(root_path, relpath):
    latest_model_subset = scipy.io.loadmat(f'{root_path}/{relpath}/latest_model_subset.mat')
    conf_mats_subsets = scipy.io.loadmat(f'{root_path}/{relpath}/conf_mats_subsets.mat')
    model_predictions = scipy.io.loadmat(f'{root_path}/{relpath}/model_predictions.mat')
    years = list(model_predictions["predictions_subset_lst"].dtype.fields.keys())
    assert set(years) == set(model_predictions["probabilities_subset_lst"].dtype.fields.keys()) and\
        set(years) == set(conf_mats_subsets["conf_mats_subsets"].dtype.fields.keys())
    years = [int(x) for x in years]
    attrition_predicted = dict()
    prob_dict = dict()
    test_data_dict = dict()
    multiple_year_idx = 0
    for i, year in enumerate(years):
        test_data = pd.read_csv(f"{root_path}/{relpath}/test_data_imputed_{year}.csv")
        test_data = test_data.loc[:, ~test_data.columns.str.contains('Unnamed')]

        curr_predictions = model_predictions["predictions_subset_lst"][0][0][i]
        curr_probabilities = model_predictions["probabilities_subset_lst"][0][0][i]
        predictions_subset = [int(val[0][0]) for val in curr_predictions]
        probs_subset = [val[0] for val in curr_probabilities]
        subset_attrition_predicted = [idx for idx, pred in list(enumerate(predictions_subset)) if pred == 1]
        curr_attritioned = test_data[test_data["Attrition"] == 1]
        curr_attritioned["Attrition Prediction"] = curr_attritioned.apply(lambda row: "Correctly Predicted" \
            if row.name in set(subset_attrition_predicted) else "Incorrectly Predicted", axis=1)
        multiple_year_idx += 1
        attrition_predicted[year] = curr_attritioned
        prob_dict[year] = probs_subset
        test_data_dict[year] = test_data
    assert prob_dict[2022] is not None, "automated subset in latest year has not been updated."
    assert test_data_dict[2022] is not None, "test data in latest year has not been updated."
    latest_subset = pd.DataFrame({"Predicted Probability": map_predictions(prob_dict[2022]),
                                  "Attrition Class": test_data_dict[2022]["Attrition"]})
    # Ensure that the probabilities and test data are the same length and have same number of rows
    overall_performance_df = generate_metrics_across_years(conf_mats_subsets["conf_mats_subsets"],
                                                           prob_dict,
                                                           test_data_dict,
                                                           years)
    return latest_subset, overall_performance_df, conf_mats_subsets, latest_model_subset, model_predictions, \
        prob_dict, test_data_dict


def create_prediction_uncertainty_chart(x, y, lower_bound, upper_bound):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='LightSkyBlue',
        line=dict(color='LightSkyBlue'),
        hoverinfo="skip",
        showlegend=False,
        opacity=0.3))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Prediction",
                             line=dict(color="lightblue", width=3)))
    fig.update_layout(title="")

    return fig


def shrink_text(txt, size):
    return f"<span style=\"font-size: {size}px;\">{txt}</span>"


def get_confusion_plot(model_name, tn, fp, fn, tp, customdata_counts):
    z = [[fn, tp],
         [tn, fp]]

    x = ["Not Attrition", "Attrition"]
    y = ["Not Attrition", "Attrition"]

    customdata_counts = np.array(customdata_counts[::-1])
    customdata_totals = [customdata_counts.sum(axis=0)[:, np.newaxis][:, 0],
                         customdata_counts.sum(axis=0)[:, np.newaxis][:, 0]]
    customdata_correct_predict = [["correctly", "incorrectly"],
                                  ["incorrectly", "correctly"]]
    customdata_actual_value = np.array([["did not attrition", "did attrition"],
                                        ["did not attrition", "did attrition"]])
    customdata_predicted = np.array([["not attritioned"] * 2, ["attritioned"] * 2])
    # set up figure
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='Blues',
                                    customdata=np.dstack((customdata_actual_value,
                                                          customdata_counts,
                                                          customdata_totals,
                                                          customdata_correct_predict,
                                                          customdata_predicted)),
                                    hovertemplate="Description: Out of %{customdata[2]}" + \
                                                  " employees<br>who %{customdata[0]}," + \
                                                  "<br>%{z:.2f}% (%{customdata[1]:.0f}) %{customdata[3]}" +
                                                  "<br>predicted as %{customdata[4]}.",
                                    name="Confusion Matrix"))

    # add title
    fig.update_layout(title_text=f'<i><b>Confusion matrix for {model_name}</b></i>',
                      yaxis=dict(title='Predicted Values'),
                      xaxis=dict(title="Actual Values"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=125, l=50, pad=0,
                                  b=50))

    # add colorbar
    fig['data'][0]['showscale'] = True
    return fig


def create_bar_chart_importances(x, y, colors, title, subtitle):
    fig = go.Figure([go.Bar(x=np.log((x / 100) + 1),
                            y=y,
                            marker=dict(color=colors),
                            customdata=x,
                            orientation='h',
                            hovertemplate="Description: A unit increase (continuous)" + \
                                          " in this<br>feature, or compared to its base class<br>(categorical)" + \
                                          ", leads" + \
                                          " to a %{customdata:.2f}% change in<br>risk of attrition",
                            name="Feature Importances")])
    fig.update_layout(title=title + f"<br><i>{subtitle}</i>")
    return fig


def create_overall_performance_chart(overall_performance_df, laz_colours, title):
    fig = go.Figure()
    colors = laz_colours[:len(overall_performance_df)]
    assert len(colors) == len(overall_performance_df), "Laz colours doesn't have enough colors"
    color_seq = [f"rgb({','.join([str(int(255 * rgb_val)) for rgb_val in palette])})" \
                 for palette in colors]
    overall_performance_df = overall_performance_df.sort_values("Year")
    for i, row in overall_performance_df.iterrows():
        row_vals = row[["Precision", "Recall", "F1 score", "AUC"]]
        fig.add_trace( \
            go.Bar(name=f"{int(row['Year'])} Performance",
                   x=["Precision", "Recall", "F1 Score", "AUC"],
                   y=row_vals,
                   marker_color=color_seq[i]))
    fig.update_layout(barmode="group",
                      title=title)
    fig.update_yaxes(tickformat=",.0%",
                     title="Estimated Performance")
    fig.update_xaxes(title="Performance Metrics")
    return fig


def create_binned_probabilities(df, predicted_probability_x_labels=["0-25%", "25-50%", "50-75%", "75-100%"],
                                barnorm="percent", barmode="stack",
                                text_addon="%", hovertext_label="Proportion"):
    grouped_lsts = df.groupby(["Predicted Probability"], as_index=False).agg(list)


    bar_dict = collections.defaultdict(list)

    for prob in predicted_probability_x_labels:
        if len(grouped_lsts[grouped_lsts["Predicted Probability"] == prob]) == 0:
            continue
        attritioned, not_attritioned = \
            get_attritioned_groups(grouped_lsts[grouped_lsts["Predicted Probability"] == prob] \
                                   .iloc[0]["Attrition Class"])
        bar_dict["Left"].append(attritioned)
        bar_dict["Stayed"].append(not_attritioned)
    x = [["(Predicted Not Likely to Attrition)",
          "(Predicted Not Likely to Attrition)",
          "(Predicted Likely to Attrition)",
          "(Predicted Likely to Attrition)"][:len(predicted_probability_x_labels)],
         predicted_probability_x_labels,

         ]

    fig = go.Figure()
    color_palettes = laz_colours

    color_seq = [f"rgb({','.join([str(int(255 * rgb_val)) for rgb_val in palette])})" \
                 for palette in color_palettes]
    fig.add_trace(go.Bar(x=x, y=bar_dict["Stayed"], name="Stayed",
                         marker_color=color_seq[8],
                         hovertemplate="<br>".join([
                             "(%{x}) " + hovertext_label + ": %{y:.0f}" + text_addon
                         ])))
    fig.add_trace(go.Bar(x=x, y=bar_dict["Left"], name="Left",
                         marker_color="darkred", hovertemplate="<br>".join([
            "(%{x}) " + hovertext_label + ": %{y:.0f}" + text_addon
        ])))
    if barnorm is not None:
        fig.update_layout(barmode=barmode,
                          barnorm=barnorm)
    else:
        fig.update_layout(barmode=barmode)
    return fig


def generate_overall_performance_table(df, title, header_color="grey",
                                       even_color="lightgrey", odd_color="white",
                                       table_height=325):
    bolded_years = [f"<b>{year}</b>" for year in \
                    sorted(df["Year"].apply(lambda x: int(x)))]
    non_year_colnames = [col for col in df if col != "Year"]
    cell_data = df[non_year_colnames].values
    cell_data = (cell_data * 100).tolist()
    cell_data = [[str(np.round(val, 1)) + "%" for val in col_lst] for col_lst in cell_data]
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Performance Metrics</b>'] + bolded_years,
            line_color='darkslategray',
            fill_color=header_color,
            align=['left', 'center'],
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[[f"<b>{val}</b>" for val in non_year_colnames]] + cell_data,
            line_color='darkslategray',
            # 2-D list of colors for alternating rows
            fill_color=[[odd_color, even_color, odd_color, even_color, odd_color] * 5],
            align=['left', 'center'],
            font=dict(color='darkslategray', size=11)
        ))
    ])
    fig.update_layout(title=title, height=table_height)

    return fig


def plot_model_outputs(model_predictions, latest_model_subset, conf_mats_subsets, latest_subset, fig_list,
                       overall_performance_df):
    # draw first row of subsets model (conf mat)
    row_height = 450
    row_width = 1400

    excludes_2020 = len(model_predictions["predictions_subset_lst"][0][0]) == 3

    for mats, mats_name, model, model_name, zeroth_row, \
            first_row, second_row, third_row, model_type, latest_df_pred_prob, \
            predicted_probability_x_labels in \
            [(conf_mats_subsets, "conf_mats_subsets", latest_model_subset,
              "latest_model_subset", "Historical Model Performance",
              "Past Years Performance (Based on Forward Feature Selection)",
              "Latest Performance (Based on Forward Feature Selection)",
              "Latest Performance/Model (Based on Forward Feature Selection)", "Forward Selection",
              latest_subset, ["0-25%", "25-50%", "50-75%", "75-100%"])]:

        index_based_on_2020_presence = (2 if excludes_2020 else 3)
        names = ["% Correctly/Incorrectly Predicted" + \
                 f"<br>{shrink_text(f'Best Model Based on {model_type}', 10)}" + \
                 f"{shrink_text(f'(tested on {name} data)', 10)}" \
                 for name in [int(k) for k in list(mats["conf_mats_subsets"].dtype.fields.keys())]]

        matrices = mats[mats_name][0][0]
        subplot_row_1 = make_subplots(rows=1, cols=index_based_on_2020_presence,
                                      subplot_titles=tuple(names[:index_based_on_2020_presence]),
                                      horizontal_spacing=(
                                          0.25 if len(model_predictions["predictions_subset_lst"][0][0]) == 3 \
                                              else 0.10), shared_yaxes=True)
        subplot_row_2 = make_subplots(rows=1, cols=2,
                                      subplot_titles=tuple(names[index_based_on_2020_presence:] + \
                                                           [
                                                               f"""Proportion of True Attrition Across Predicted Probabilities""" + \
                                                               f"""<br>{shrink_text(f'Based on {model_type}', 12)}"""]),
                                      horizontal_spacing=0.25)
        subplot_row_3 = make_subplots(rows=1, cols=2, horizontal_spacing=0.25,
                                      subplot_titles=["Counts of Attrition Across Predicted Probabilities",
                                                      f"{shrink_text(f'Model Feature Importances Based on {model_type} (trained on 2021 data)', 12)}<br>" +
                                                      f"{shrink_text('Attrition Risk from Sex Relative to Female', 10)}" +
                                                      f", {shrink_text('Corporate Title (Associate)', 10)}" + \
                                                      f", {shrink_text('Comp Tier (Tier <= 2)', 10)}"])
        subplot_row_0 = generate_overall_performance_table(overall_performance_df,
                                                           title="Overall Performance of Model Across Historical Years")

        for j in range(1, len(matrices) + 1):
            #         if j == 2:
            #             continue
            cm = matrices[j - 1]
            counts = [cm[1, :], cm[0, :]]
            cm = np.transpose(np.transpose(cm).astype('float') / cm.sum(axis=0)[:, np.newaxis])
            model_mat_name = mats_name.split('_')[-1]
            model_mat_name = model_mat_name[0].upper() + model_mat_name[1:]
            fig = get_confusion_plot(f"Regularized Logistic ({model_mat_name})",
                                     *(np.round(100 * np.append(cm[1, :], cm[0, :]), 2)),
                                     counts)
            if j <= index_based_on_2020_presence:
                for trace in range(len(fig["data"])):
                    subplot_row_1.add_trace(fig["data"][trace], row=1, col=j)

                subplot_row_1.update_xaxes(title="Actual Values", row=1, col=j)
                subplot_row_1.update_yaxes(title="Predicted Values",
                                           row=1, col=j)
            else:
                for trace in range(len(fig["data"])):
                    subplot_row_2.add_trace(fig["data"][trace], row=1, col=j - index_based_on_2020_presence)

                subplot_row_2.update_xaxes(title="Actual Values", row=1, col=j - index_based_on_2020_presence)
                subplot_row_2.update_yaxes(title="Predicted Values",
                                           row=1, col=j)
                subplot_row_2.update_traces(showscale=False)
        #### Feature Importance Bar Charts ####
        names = [' '.join(wordninja.split(' '.join(name[0][0].replace("T.1", "") \
                                                   .replace("X", "") \
                                                   .replace(".", " ") \
                                                   .replace("_", " ") \
                                                   .split()))).replace(" s ", "s ")\
                                         for name in model["names"][0][0][-1]][1:]

        if mats_name == "conf_mats_visuals":
            for i in range(len(names)):
                if ("self exceeds 360 manager categorical" in names[i]):
                    names[i] = "Self > 360, Manager (Categorical)"
        for i in range(len(names)):
            if ("YO Y TC Ch g" in names[i]):
                names[i] = "YOY Total Compensation % Change"
            elif ("latest per c change women by dept" in names[i]):
                names[i] = "Latest % Change in Women by Department"
            elif ("latest per c women by dept" in names[i]):
                names[i] = "Latest % of Women by Department"
            elif ("latest per c change women by supervisor" in names[i]):
                names[i] = "Latest % Change in Women by Supervisor"

        model_suffix = model_name.split('_')[-1]
        model_suffix = model_suffix[0].upper() + model_suffix[1:]

        # coefficients from the most recent year
        coefs = [coef_lst[0] for coef_lst in model[model_name][0][0][-1]][1:]
        coef_dict = dict()
        for k in range(len(coefs)):
            coef_dict[names[k]] = coefs[k]

        sorted_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda x: abs(x[1]),
                                               reverse=False)}
        new_names, new_coefs = [], []
        for name, coef in sorted_dict.items():
            new_names.append(name)
            new_coefs.append(coef)
        names, coefs = new_names, new_coefs
        coefs = 100*(np.exp(np.array(coefs))-1)
        colors = []
        for coef in coefs:
            colors.append("firebrick" if coef < 0 else "darkblue")

        fig_predicted_attrition = create_binned_probabilities(latest_df_pred_prob,
                                                              predicted_probability_x_labels=predicted_probability_x_labels)

        for trace in range(len(fig_predicted_attrition["data"])):
            subplot_row_2.add_trace(fig_predicted_attrition["data"][trace], row=1,
                                    col=j - (index_based_on_2020_presence - 1))
        subplot_row_2.update_yaxes(title="Proportion of Attritioned/Not Attritioned", row=1,
                                   col=j - (index_based_on_2020_presence - 1))

        fig_counts = create_binned_probabilities(latest_df_pred_prob,
                                                 predicted_probability_x_labels=predicted_probability_x_labels,
                                                 barnorm=None, barmode="group",
                                                 text_addon="", hovertext_label="Count")
        for trace in range(len(fig_counts["data"])):
            subplot_row_3.add_trace(fig_counts["data"][trace], row=1, col=1)
        subplot_row_3.update_yaxes(title="Counts of Attritioned/Not Attritioned", row=1, col=1)
        # print(np.array(coefs))
        fig_bar = create_bar_chart_importances(np.array(coefs), names, colors,
                                               title=f"Regularized Logistic Model {model_suffix}" + \
                                                     " -  Relative Risks of Attrition in Features",
                                               subtitle="Attrition Risk from Sex Relative to Female," +
                                                        " Corporate Title Relative to Associate")
        for trace in range(len(fig_bar["data"])):
            subplot_row_3.add_trace(fig_bar["data"][trace], row=1, col=2)

        subplot_row_1.update_layout(width=row_width,
                                    height=row_height,
                                    showlegend=False,
                                    font=dict(size=10),
                                    margin=dict(l=0, b=0))
        subplot_row_1.update_traces(showscale=False)
        fig_list[zeroth_row] = subplot_row_0
        fig_list[first_row] = subplot_row_1

        # draw second row of subset models (conf mat of latest year)
        subplot_row_2.update_layout(barmode="stack",
                                    barnorm="percent", width=row_width,
                                    height=row_height + 100,
                                    showlegend=False,
                                    font=dict(size=10),
                                    margin=dict(l=0, b=0))

        subplot_row_3.update_layout(width=row_width,
                                    height=row_height,
                                    showlegend=False,
                                    font=dict(size=10),
                                    margin=dict(l=0, b=0))
        fig_list[second_row] = subplot_row_2
        fig_list[third_row] = subplot_row_3
    return fig_list


def _write_plotly_go_to_html_model_results(fig_list, root_path, output_filepath, title, model_name):
    """
    Write list of plotly graph objects to html
    """
    with open(f"{root_path}/{output_filepath}", "w") as f:
        f.write(f"<h1>{title}</h1>")
        f.write(f"<h4> [Filtered on] {inclusion_filter}<br><br>{exclusion_filter} </h4>")
        f.write("<hr>")
        f.write("<h1>Model Information</h1>")

        f.write(f"<h2> Name: {model_name}</h2>")
        f.write("<hr>")
        for figure in fig_list:
            f.write(f"<h1>{figure}</h1>")
            f.write(fig_list[figure].to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("<hr>")
        html = codecs.open(f"{root_path}/{output_filepath}", "r", "utf-8").read()

    return html


# Shapley value plots

def add_predictions_and_risks(prob_lst, test_data_lst, col_order=FA_COLUMN_ORDER,
                              rename_dict=FA_ALL_RENAME_DICT):
    # Assume the exclusion of 2020.
    assert len(prob_lst) == len(test_data_lst), \
        "Predicted Probabilities are not the same length as the test data"
    data_lst = []
    # Assuming that we have 2018
    for year in test_data_lst:
        curr_data = test_data_lst[year]
        curr_data["Risk Probability"] = prob_lst[year]
        lower, upper = np.quantile(prob_lst[year], 0.25), np.quantile(prob_lst[year], 0.75)
        curr_data["Relative Risk"] = curr_data["Risk Probability"].apply(lambda x: "Low Risk" if x < lower \
            else "Medium Risk" if x < upper \
            else "High Risk")
        curr_data["25th Percentile Risk (Below is Low)"] = lower
        curr_data["75th Percentile Risk (Above is High)"] = upper
        data_lst.append(curr_data)
    data = pd.concat(data_lst)
    assert all(col in data.columns for col in ["Year", "Attrition"]), "Data does not contain the Year column"

    assert set(data["Year"].unique()) == {2018, 2019, 2021, 2022}
    data = data.rename(columns=rename_dict)[col_order + ["Year", "Attrition", "Risk Probability",
                                                         "25th Percentile Risk (Below is Low)",
                                                         "75th Percentile Risk (Above is High)",
                                                         "Relative Risk"]].reset_index(drop=True)
    return data


def renaming_vars(rename_dict, names):
    lst = []
    for name in names:
        if name in rename_dict:
            lst.append(rename_dict[name])
        else:
            matches = [col for col in rename_dict if col in name]
            if len(matches) == 0:
                lst.append(name)
                continue
            old_col = matches[0]
            new_col = rename_dict[old_col]
            lst.append(name.replace(old_col, new_col))
    return lst


#####################
# Probability-space #
#####################

def replace_strings(L):
    return [1 if type(val) == str else 0 if pd.isna(val) else val for val in L]


# logistic function being the sigmoid function.
def sigmoid(L):
    return 1 / (1 + np.exp(-1 * np.sum(L)))


def map_col_to_coef(coefs, row, col):
    return coefs[col] if col in coefs else \
        coefs[f"{col}{row[col]}"] \
            if f"{col}{row[col]}" in coefs else 0


def map_coefs(row, coefs, relevant_cols, idx_lst):
    return [map_col_to_coef(coefs, row, relevant_cols[j]) for j in idx_lst]


def compute_individual_contributions(row, coef_dicts, reverse_rename_dict,
                                     relevant_cols, categorical_cols):
    intercept = coef_dicts[row["Year"]]["(Intercept)"]
    coefs = coef_dicts[row["Year"]]
    contributions = []
    row_coefs = map_coefs(row, coefs, relevant_cols, range(len(relevant_cols)))
    standardized_feature_vals = row[relevant_cols].tolist()
    standardized_feature_vals = replace_strings(standardized_feature_vals)

    prob = sigmoid([intercept] + list(np.multiply(row_coefs, standardized_feature_vals)))

    marginal_effect_contributions = []

    for i, col in enumerate(relevant_cols):
        # Approach 1: Exclusion of individual feature - doesn't condition on other features
        # Approach 2: Marginal effects by holding others constant using averages - only an estimate.
        # Approach 3 (Current): Don't set the other features to be constant - measures global contribution
        # we instead hold the other features constant to what they were before.
        if col in categorical_cols:
            dummy_var_name = f"{col}{row[col]}"
            if dummy_var_name not in coefs:
                # base class
                contributions.append(0)
                continue

        curr_feature_val = standardized_feature_vals[i]
        contributions.append(row_coefs[i] * curr_feature_val)
    return contributions, intercept


def obtain_contributions(row):
    row_shap_vals = row["values"]
    row_base_vals = row["base_values"]
    probability = sigmoid([row_base_vals + np.sum(row_shap_vals)])
    idx_shap = list(enumerate(row_shap_vals))
    shap_len = len(idx_shap)
    base_shap = row_base_vals
    contributions = [0] * shap_len
    idx_shap_sorted = sorted(idx_shap, key=lambda x: abs(x[1]))
    contributions = [0] * len(idx_shap)

    curr_idx = 0
    exp_prob = sigmoid([base_shap])
    prev_shap = base_shap
    while curr_idx < shap_len:
        shap_idx, shap_val = idx_shap_sorted[curr_idx]
        next_shap = prev_shap + shap_val
        contributions[shap_idx] = sigmoid([next_shap]) - sigmoid([prev_shap])

        prev_shap = next_shap
        curr_idx += 1
    return contributions + [exp_prob, exp_prob + np.sum(contributions)]


def add_feature_contributions(fa_data, coef_dicts, col_order=FA_COLUMN_ORDER[1:],
                              reverse_rename_dict=FA_REVERSE_RENAME_DICT,
                              categorical_cols=FA_CATEGORICALS):
    marginal_effects = fa_data.copy()
    relevant_cols = [col for col in marginal_effects.columns if col not in {"Employee ID", 'Year',
                                                                            'Attrition', 'Risk Probability',
                                                                            '25th Percentile Risk (Below is Low)',
                                                                            '75th Percentile Risk (Above is High)',
                                                                            'Relative Risk'}]
    marginal_effects[["values", "base_values"]] = marginal_effects.apply(lambda row:
                                                                         compute_individual_contributions(row,
                                                                                                          coef_dicts,
                                                                                                          reverse_rename_dict,
                                                                                                          relevant_cols,
                                                                                                          categorical_cols),
                                                                         axis=1,
                                                                         result_type="expand")
    marginal_effects[relevant_cols + ["Expected Attrition Probability",
                                      "Risk Probability"]] = marginal_effects[
        ["values", "base_values", "Risk Probability"]] \
        .apply(obtain_contributions,
               axis=1,
               result_type="expand")
    return marginal_effects


def create_coef_dicts(latest_model_subset, rename_dict=FA_ALL_RENAME_DICT):
    years = list(latest_model_subset["latest_model_subset"].dtype.fields.keys())
    years = [int(x) for x in years]
    coefs = latest_model_subset["latest_model_subset"][0][0]
    coefs = [[coef[0] for coef in coef_lst] for coef_lst in coefs]
    names = latest_model_subset["names"][0][0]
    names = [[name[0][0] for name in name_lst] for name_lst in names]
    assert len(coefs) == len(names), "Not equal length coefficients and variable names"
    coef_dicts = [dict(zip(renaming_vars(rename_dict, names[i]), coefs[i])) for i in range(len(coefs))]

    coef_dict_fin = dict()
    for i, year in enumerate(years):
        coef_dict_fin[year] = coef_dicts[i]
    return coef_dict_fin


def add_dfs(dfs, names, output_name):
    writer = pd.ExcelWriter(output_name)
    assert len(dfs) == len(names), "Not equal length dataframes and names"
    for i, df in enumerate(dfs):
        df.to_excel(writer, sheet_name=names[i])
    writer.save()


if __name__ == "__main__":
    args = [("All", "output/Flight Risk Top Model - All.html",
             "Entire Employee Population",
             "All Logistic With Automated Features Excluding 2020, " +
             "Included % Women Factors Chosen With Highest F1-Score in 2022"),
            ("Female", "output/Flight Risk Top Model - Female.html",
             "Female Employee Population",
             "Female Logistic With Automated Features " +
             "Trained on Past 4 Years Chosen With Highest F1-Score in 2022"),
            ("Male", "output/Flight Risk Top Model - Male.html",
             "Male Employee Population",
             "Male Logistic Based on Flight Risk EDA " +
             "Chosen With Highest F1-Score in 2022"),
            ("VP", "output/Flight Risk Top Model - VP.html",
             "VP Employee Population",
             "VP Logistic With Automated Features" +
             " Trained on Past 4 Years Chosen With Highest F1-Score in 2022"),
            ("Associate", "output/Flight Risk Top Model - Associate.html",
             "Associate Employee Population",
             "Associate Logistic With Automated Features Excluding 2020" +
             " Chosen With Highest F1-Score in 2022"),
            ("Director", "output/Flight Risk Top Model - Director.html",
             "Director Employee Population",
             "Director Logistic With Automated Features Excluding 2020" +
             " Chosen With Highest F1-Score in 2022")]
    inclusion_filter = "<br><br>Include: Financial Advisory, Revenue Producing, post-2017, US,<br>" + \
                       " Canada, France, UK, Italy, Germany, Spain, Sweden, Netherlands, Belgium, Switzerland."
    exclusion_filter = "Exclude: interns, analysts, senior advisors, managing directors, associate 0's,<br>" + \
                       "employees hired in same year as headcount year or the same year as<br>" + \
                       "their latest comp tier year, Private Capital Advisory (PCA), Investor<br>" + \
                       "Relations Advisory, Venture and Growth Banking, Data Analytics Group<br>"
    root_path = extract_root_fpath("llama")
    for (title_folder, output_path, title, model_name) in args:
        for bus_unit in ["FA"]:
            # TODO: Hardcoded All title folder for now
            if title_folder != "All":
                continue
            print(f"data/figs/{title_folder}/{bus_unit}/")
            latest_subset, overall_performance_df, \
                conf_mats_subsets, latest_model_subset, \
                model_predictions, prob_lst, test_data_lst = read_mat_files(root_path,
                                                                            relpath=f"data/figs/{title_folder}/{bus_unit}")
            fa_data = add_predictions_and_risks(prob_lst, test_data_lst)

            coef_dicts = create_coef_dicts(latest_model_subset)
            contributions_df = add_feature_contributions(fa_data, coef_dicts)

            add_dfs([fa_data, contributions_df],
                    ["Predictors", "Model Contributions"],
                    f"{root_path}/output/HR Flight Risk Predictions V05.xlsx")

            fig_list = plot_model_outputs(model_predictions, latest_model_subset, conf_mats_subsets, latest_subset,
                                          dict(), overall_performance_df)
            html = _write_plotly_go_to_html_model_results(fig_list, root_path, output_path, title, model_name)
