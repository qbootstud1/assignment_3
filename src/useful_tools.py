import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import math
import numpy as np


def df_features_preanalysis(dataframe: pd.DataFrame):
    """
    Pre-analysis of each of the dataframe's features and labels. It includes the counting of the rows and columns,
     NaN values, and repeated elements. For each features, it also includes data type and unique values analysis.
    :param dataframe: input dataframe to analyze
    :return: print of all the computed values and dataframe's characteristics.
    """
    n_rows_repeated = dataframe[dataframe.duplicated(keep='first')].shape[0]
    null_values = sum(dataframe.isnull().sum())
    n_rows_original = dataframe.shape[0]
    n_columns_original = dataframe.shape[1]
    print('─' * 150)
    print('GENERAL INFO')
    print('─' * 150)
    print(f'The original dataset has {n_rows_original:,} rows and {n_columns_original} columns, with a total of '
        f'{null_values} NaN values and {n_rows_repeated} rows repeated')
    print('· Statistical info about the numerical features')
    print(display(dataframe.describe()))
    print('─' * 150)
    print('FEATURES INFO')
    print('─' * 150)

    for i, col in enumerate(dataframe.columns):
        print(f'------------------------------------|   FEATURE {i}: {col}   |------------------------------------')
        print(dataframe[col].apply(type).value_counts())
        print('')
        if len(dataframe[col].unique()) < 10:
            print(
                f'In this column there are {len(dataframe[col].unique())} unique values and'
                f' {dataframe[col].isnull().sum()} NaN values:')
            print(dataframe[col].value_counts(dropna=False))
            print()
        else:
            print(f'The first ten unique elements are:')
            cnt = 0
            for element in dataframe[col].unique():
                cnt += 1
                if cnt < 10:
                    print(element)
            print('...')
            print(
                f'In this column there are {len(dataframe[col].unique())} unique values and'
                f' {dataframe[col].isnull().sum()} NaN values\n')
            
            
def features_visualization(dataframe: pd.DataFrame, label_name: str, label_values: Tuple,
                           save_figure: bool = False, num_plot_columns: int = 3, **kwargs) -> None:
    """

    :param dataframe:
    :param label_name:
    :param label_values:
    :param save_figure:
    :param kwargs:
    :return:
    """
    number_cols = num_plot_columns
    number_rows = math.ceil(len(dataframe.columns)/number_cols)
    fig, ax = plt.subplots(figsize=(13, 3*number_rows))

    for n, col in enumerate(dataframe.columns[:-1]):
        # Add subplot
        plt.subplot(number_rows, number_cols, n + 1)

        # Define bins
        bins = dataframe[col].unique().shape[0]

        # Plot entries
        line_width = 1.9
        color_vec = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        for order, value in enumerate(label_values):
            line_width -= 0.2
            dataframe[col].loc[dataframe[label_name] == value].hist(density=True, histtype='step', bins=bins,
                                                                    color=color_vec[order], linewidth=line_width,
                                                                    alpha=1, grid=False, ax=plt.gca(), label=f'Label: {value}')

        # Options
        plt.title(col)
        plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
        plt.legend()

    # Space plots nicely
    plt.tight_layout()

    # Save and close
    if save_figure:
        if kwargs:
            fig.savefig(kwargs['path'] + f'/results/visualize_inputs/' + kwargs['plot_name'] + '.pdf',
                        bbox_inches='tight', pad_inches=0.05)
        else:
            print('ERROR: if save_plots is set True, you need to provide the arguments path=str and plot_name=str')
            
    plt.show()
    

def features_correlations(dataframe: pd.DataFrame, label_name: str):
    # Take all signal and a random subsample of background
    all_nok = dataframe.loc[dataframe[label_name] == 1]
    all_ok = dataframe.loc[dataframe[label_name] == 0]#.sample(n=len(all_nok), random_state=123)

    # Join both dataframes
    data_balanced_all = pd.concat([all_nok, all_ok], axis=0)

    for col in data_balanced_all.columns[:-1]:
        data_balanced_all[col] = data_balanced_all[col].astype('category').cat.codes
    # Plot the correlation matrix
    fig = plt.figure(figsize=(11, 9))
    # Plot matrix
    plt.matshow(data_balanced_all.corr(), cmap=plt.cm.jet, vmin=-1, vmax=1, fignum=fig.number)
    for (i, j), z in np.ndenumerate(data_balanced_all.corr()):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    # Options
    plt.title('Correlation Matrix', fontsize=16)
    plt.xticks(range(data_balanced_all.shape[1]), data_balanced_all.columns, fontsize=14, rotation=90)
    plt.yticks(range(data_balanced_all.shape[1]), data_balanced_all.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    plt.show()
    

def plot_model_comparison(rf_score_train, dt_score_train, svc_score_train, rf_score_test, dt_score_test, svc_score_test):
    # Intitialize figure with two plots

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(14)
    fig.set_facecolor('white')

    # First plot
    ## set bar size
    barWidth = 0.2

    ## Set position of bar on X axis
    r1 = np.arange(len(rf_score_train))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    ## Make the plot
    ax1.bar(r1, rf_score_train, width=barWidth, edgecolor='white', label='Random Forest')
    ax1.bar(r2, dt_score_train, width=barWidth, edgecolor='white', label='Decision Tree')
    ax1.bar(r3, svc_score_train, width=barWidth, edgecolor='white', label='SVC')

    ## Configure x and y axis
    ax1.set_xlabel('Metrics', fontweight='bold')
    labels = ['Precision', 'Recall', 'F1']
    ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(rf_score_train))], )
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_ylim(0, 1)

    ## Create legend & title
    ax1.set_title('Evaluation Metrics - Training', fontsize=14, fontweight='bold')
    ax1.legend()

    # Second plot
    ## Set position of bar on X axis
    r1 = np.arange(len(rf_score_test))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    ## Make the plot
    ax2.bar(r1, rf_score_test, width=barWidth, edgecolor='white', label='Random Forest')
    ax2.bar(r2, dt_score_test, width=barWidth, edgecolor='white', label='Decision Tree')
    ax2.bar(r3, svc_score_test, width=barWidth, edgecolor='white', label='SVC')

    ## Configure x and y axis
    ax2.set_xlabel('Metrics', fontweight='bold')
    labels = ['Precision', 'Recall', 'F1']
    ax2.set_xticks([r + (barWidth * 1.5) for r in range(len(rf_score_test))], )
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_ylim(0, 1)

    ## Create legend & title
    ax2.set_title('Evaluation Metrics - Test', fontsize=14, fontweight='bold')
    ax2.legend()

    plt.show()
