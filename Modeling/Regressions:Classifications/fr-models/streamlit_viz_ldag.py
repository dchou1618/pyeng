import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import re

base_dir = (os.path.join(os.path.dirname(__file__), '../').replace("\\", "/")
            .replace("src/../", "").replace("streamlit_app/../", "")
            )
sys.path.insert(1, base_dir)

import src.Lazard_looknfeel_v2 as laz

# Discrete LAZ colour palette
laz_colours = []
for colour in ['cephalopod ink', 'moonstone', 'moonstone_40', 'gold', 'gold_40']:
    laz_colours += [laz.primary_palette[colour]]
for colour in ['hunter', 'moss', 'olive', 'ocean', 'bark', 'tangerine', 'daffodil', 'lilac', 'chestnut']:
    laz_colours += [laz.extended_palette[colour]]
laz_colours = ['rgb' + str(tuple([int(x * 255) for x in palette])) for palette in laz_colours]


def palette_to_str(laz_palette):
    """
    Helper function to convert list of RGB codes laz_palette to str format
    :param laz_palette: lst
    :return: str
    """
    laz_str = re.sub("\[", "(", str([int(255 * x) for x in laz_palette]))
    laz_str = re.sub("\]", ")", laz_str)
    return f'rgb{laz_str}'


def ldag_plot_heatmap(df, plot_title):
    """
    Plot pivoted df where index is x axis and columns are y axis. Use values for colour intensity
    :param df: pd.DataFrame
    :param plot_title: str
    :return: plotly.graph_objects.Figure
    """
    heatmap_args = {'hoverongaps': False,
                    'colorscale': [(0, "white"),
                                   (1, palette_to_str(laz.secondary_palette['monterey']))
                                   ]
                    }

    # Transpose it first so values also get transposed
    df = df.transpose()
    fig = go.Figure(data=go.Heatmap(z=df.values,
                                    y=df.index.to_list(),
                                    x=df.columns.to_list(),
                                    **heatmap_args),
                    layout={'title': plot_title,
                            'yaxis': {'autorange': 'reversed'}}
                    )
    return fig


def ldag_plot_line(df, x_col, y_col, colour_col, plot_title):
    """
    Plot line chart using data in df with x_col on x-axis, y_col on y-axis, and colour corresponding to colour_col with
    title plot_title
    :param df: pd.DataFrame
    :param x_col: str
    :param y_col: str
    :param colour_col: str
    :param plot_title: str
    :return: plotly.graph_objects.Figure
    """
    fig = px.line(df,
                  x=x_col,
                  y=y_col,
                  color=colour_col,
                  color_discrete_sequence=laz_colours,
                  title=plot_title,
                  markers=True
                  )

    fig.update_layout(xaxis_title='', template='plotly_white')

    return fig


def ldag_plot_bar(df, x_col, y_col, colour_col, plot_title, barmode='stack'):
    """
    Plot stacked bar chart using data in df with x_col on x-axis, y_col on y-axis, and colour corresponding to
    colour_col with title plot_title
    :param df: pd.DataFrame
    :param x_col: str
    :param y_col: str
    :param colour_col: str
    :param plot_title: str
    :param barmode: str. 'stack' for colours to be on top of one another, 'group' to be next to one another
    :return: plotly.graph_objects.Figure
    """
    fig = px.bar(df,
                 x=x_col,
                 y=y_col,
                 color=colour_col,
                 barmode=barmode,
                 color_discrete_sequence=laz_colours,
                 title=plot_title
                 )

    # Ensure order of legend colours matches order of stacked bar colours
    if barmode == 'stack':
        fig.update_layout(legend_traceorder='reversed')

    fig.update_layout(xaxis_title='', template='plotly_white')

    return fig
