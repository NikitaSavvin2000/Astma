import os

import pandas as pd
from matplotlib.ticker import MaxNLocator
from translate import Translator
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

translator = Translator(to_lang='en', from_lang='ru')


def translate_column_names(df):
    new_columns = {}
    for column in df.columns:
        # получаем перевод и добавляем его в новый словарь
        translated_text = translator.translate(column)
        new_columns[column] = translated_text
    # переименовываем столбцы в новом словаре
    return df.rename(columns=new_columns)


'''def plot_distributions(df, folder_name):
    
    In this function you must input dataframe 'df' and 'folder_name' there will saving you results
    Your results you could find in result/'folder_name'
    Into df need be only float or int values
    
    if not os.path.isdir(os.path.join(os.getcwd(), 'result')):
        os.mkdir(os.path.join(os.getcwd(), 'result'))

    folder_path = os.path.join(os.getcwd(), 'result', folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    for col in df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        bin_width = (np.max(df[col].dropna()) - np.min(df[col].dropna())) / 20
        sns.histplot(data=df[col].dropna(), bins=np.arange(np.min(df[col]), np.max(df[col]), bin_width), kde=True,
                     label='KDE - kernel density estimation', ax=ax1)

        mu, std = norm.fit(df[col].dropna())
        gauss_x = np.linspace(np.min(df[col]), np.max(df[col]), 100)
        gauss_y = norm.pdf(gauss_x, mu, std)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Gaussian')
        ax2.plot(gauss_x, gauss_y, color="#FFC300", label='Gaussian distribution')

        line1 = ax1.axvline(x=np.mean(df[col]), color='red', linestyle='--', label='Mean')
        line2 = ax1.axvline(x=np.mean(df[col]) - np.std(df[col]), color='green', linestyle='--',
                            label='Mean -+ Std. Dev.')
        ax1.axvline(x=np.mean(df[col]) + np.std(df[col]), color='green', linestyle='--')
        mean_legend = plt.Line2D([], [], color='white', label=f"Arithmetic mean = {np.mean(df[col]):.2f}")
        std_legend = plt.Line2D([], [], color='white', label=f"Standard deviation = {np.std(df[col]):.2f}")
        var_legend = plt.Line2D([], [], color='white', label=f"Dispersion = {np.var(df[col]):.2f}")
        dist_legend = plt.Line2D([], [], label='KDE - kernel density estimation')
        Gaussian_legend = plt.Line2D([], [], color="#FFC300", label='Gaussian distribution')
        ax1.legend(handles=[Gaussian_legend, dist_legend, line1, line2, mean_legend, std_legend, var_legend],
                   title='Statistics')
        plt.title(f"Distribution graph for - '{col}'", fontsize=13, y=1.02)
        ax1.grid(True, linewidth=0.75, color='black')
        ax2.grid(True, linewidth=0.5, linestyle='--')
        plt.savefig(os.path.join(folder_path, f"{col}.png"))
        plt.clf()'''


def calculate_tick_parameters(data):

    unique_values = np.array(data.unique())
    unique_values = unique_values.tolist()
    unique_values_round = [round(x) for x in unique_values]
    unique_values_round = set(unique_values_round)
    if len(unique_values_round) > 25:
        unique_values = unique_values_round
    print(unique_values)
    num_unique_values = len(unique_values)
    print(unique_values)

    if num_unique_values <= 25:
        tick_values = unique_values
    else:
        max_value = 1
        for number in unique_values:
            while (max_value * number) % 1 != 0:
                max_value *= 10
        # Находим минимальное и максимальное число в списке
        minimum = (min(unique_values)) * max_value
        maximum = (max(unique_values)) * max_value
        minimum = 5 * (minimum // 5)
        maximum = 5 * ((maximum + 5) // 5)
        count = (maximum - minimum) / 23
        step = (5 * round(count / 5))
        if step == 0:
            step = round(5 * (count / 5))
        tick_values = [(round(minimum + i * step)) / max_value for i in range(24)]
        while tick_values[-1] < max(unique_values):
            tick_values.append(tick_values[-1] + (step)/max_value)
        while tick_values[-2] >= max(unique_values) and tick_values[-1] >= max(unique_values):
            tick_values.pop(-1)
    print(tick_values)
    tick_values.sort()
    print(tick_values)
    return tick_values


def dict_bins(df):
    result_dict = {}
    for col in df.columns:
        data = df[col].dropna()
        result_dict[col] = calculate_tick_parameters(data)
    print(result_dict)
    return result_dict

def plot_distributions(df, bins, folder_name):
    '''
    In this function you must input dataframe 'df' and 'folder_name' there will saving you results
    Your results you could find in result/'folder_name'
    Into df need be only float or int values
    '''
    if not os.path.isdir(os.path.join(os.getcwd(), 'result')):
        os.mkdir(os.path.join(os.getcwd(), 'result'))

    folder_path = os.path.join(os.getcwd(), 'result', folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    # Assuming df is your DataFrame and calculate_tick_parameters is a function you have defined

    for col in df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        data = df[col].dropna()
        print(col)
        # Determine bin width
        x_min, x_max = data.min(), data.max()
        num_ticks = list(bins[col])
        print(num_ticks)
        tick_width = num_ticks[1] - num_ticks[0]
        bin_width = tick_width  # Set bin width to be equal to the tick width
        print(tick_width)
        # Adjust bin edges
        bin_edges = np.arange(num_ticks[0], num_ticks[-1] + bin_width, bin_width)
        print(f'bins {bin_edges}')

        if len(bins) == 0:
            continue

        ax1.set_xticks(num_ticks)
        ax1.tick_params(axis='x', labelrotation=45, labelsize=10)
        sns.histplot(data=data, bins=bin_edges, kde=True, label='KDE - kernel density estimation', ax=ax1)

        mu, std = norm.fit(data)
        gauss_x = np.linspace(x_min, x_max, 100)
        gauss_y = norm.pdf(gauss_x, mu, std)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Gaussian')
        ax2.plot(gauss_x, gauss_y, color="#FFC300", label='Gaussian distribution')

        line1 = ax1.axvline(x=np.mean(data), color='red', linestyle='--', label='Mean')
        line2 = ax1.axvline(x=np.mean(data) - np.std(data), color='green', linestyle='--', label='Mean -+ Std. Dev.')
        ax1.axvline(x=np.mean(data) + np.std(data), color='green', linestyle='--')
        mean_legend = plt.Line2D([], [], color='white', label=f"Arithmetic mean = {np.mean(data):.2f}")
        std_legend = plt.Line2D([], [], color='white', label=f"Standard deviation = {np.std(data):.2f}")
        var_legend = plt.Line2D([], [], color='white', label=f"Dispersion = {np.var(data):.2f}")
        dist_legend = plt.Line2D([], [], label='KDE - kernel density estimation')
        Gaussian_legend = plt.Line2D([], [], color="#FFC300", label='Gaussian distribution')
        ax1.legend(handles=[Gaussian_legend, dist_legend, line1, line2, mean_legend, std_legend, var_legend],
                   title='Statistics')
        plt.title(f"Distribution graph for - '{col}'", fontsize=13, y=1.02)


        ax1.grid(True, linewidth=0.4, color='black')
        ax2.grid(True, linewidth=0.5, linestyle='--')
        plt.savefig(os.path.join(folder_path, f"{col}.png"))
        plt.clf()


def correlation_matrix(df, folder_name):
    if not os.path.isdir(os.path.join(os.getcwd(), 'result')):
        os.mkdir(os.path.join(os.getcwd(), 'result'))

    folder_path = os.path.join(os.getcwd(), 'result', folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    # Создаем словарь с номерами колонок
    col_names = dict(zip(df.columns, map(str, range(1, len(df.columns) + 1))))
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(13, 6))

    # Устанавливаем метки на оси x и y
    # Создаем тепловую карту
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, ax=ax)

    ax.set_xticklabels([f"{col_names[col]} " for col in df.columns], rotation=0)
    ax.set_yticklabels([f"{col_names[col]} " for col in df.columns])

    ax.set_title("Correlation Matrix", fontsize=13, y=1.02)
    fig.subplots_adjust(right=0.75)

    text_ax = fig.add_axes([0.73, 0.072, 0.25, 0.75], frameon=False)
    text_ax.axis('off')
    text_ax.text(
        0, 0.5, '\n'.join([f'{v} - {k}' for k, v in col_names.items()]),
        fontsize=10, linespacing=2.5,
        bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', pad=1)
    )

    plt.savefig(os.path.join(folder_path, f"Correlation Matrix.png"))
