# Data processing  
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualization  
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Mathematics  
# -----------------------------------------------------------------------
import math


def checker(df, col):
    """
    Provides summary statistics and insights about a specific column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to analyze.
    - col (str): The name of the column to check.

    Returns:
    - None: The function prints the summary statistics directly.
    """

    print(f"Number of entries: {df.shape[0]}.")
    print(f"Number of {col} distinct entries: {df[col].nunique()}.")
    print(f"Number of {col} duplicated: {df[col].duplicated().sum()}.")
    print(f"Number of {col} null: {df[col].isna().sum()}.")


def plot_numeric_distribution(df, first, last, col, n=1, size = (10, 5), rotation=45):
    """
    Plots the distribution of numeric values in a specified column within a given range, using aligned bins.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - first (int or float): The lower bound of the range to plot.
    - last (int or float): The upper bound of the range to plot.
    - col (str): The name of the column to analyze.
    - n (int, optional): The bin width for the histogram. Defaults to 1.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (10, 5).
    - rotation (int, optional): The rotation angle for x-axis labels. Defaults to 45.

    Raises:
    - ValueError: If `n` is not positive or if `first` is not less than `last`.

    Returns:
    - None: Displays a histogram of the numeric distribution within the specified range.
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Bin width (n) must be a positive integer.")
    if first >= last:
        raise ValueError("'first' must be less than 'last'.")
    
    # Define the bin edges to align with ticks
    bin_edges = np.arange(first, last + n, n)
    
    # Filter the data
    filtered_data = df[df[col].between(first, last)][col]
    
    if filtered_data.empty:
        print(f"No data available for the range {first} to {last}.")
        return

    # Set dynamic figure size based on the range
    plt.figure(figsize=size)
    
    # Create the histogram with aligned bins
    sns.histplot(filtered_data, bins=bin_edges, kde=False, color="skyblue", edgecolor="black")
    
    # Add title and labels
    plt.title(f"Distribution of {col} ({first}–{last})")
    plt.xlabel("")
    plt.ylabel("Frequency")
    
    # Set x-ticks to align with bins
    plt.xticks(bin_edges, rotation=rotation)
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_categoric_distribution(df, col, size = (8, 4), color='mako', rotation=45, order=True):
    """
    Plots the distribution of a categorical column, showing the count of each unique value.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the categorical column to analyze.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (8, 4).
    - color (str, optional): The color palette for the bars. Defaults to 'mako'.
    - rotation (int, optional): The rotation angle for x-axis labels. Defaults to 45.

    Returns:
    - None: Displays a bar plot showing the counts of each unique value in the categorical column.
    """

    if order:
        order=df[col].value_counts().index

    else: 
        order=None

    plt.figure(figsize=size)

    sns.countplot(
        x=col, 
        data=df[col].reset_index(),  
        palette=color, 
        order=order,
        edgecolor="black"
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel("")
    plt.ylabel("Number of registrations")

    plt.xticks(rotation=rotation)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, size = (5, 5)):
    """
    Plots the correlation matrix of numeric columns in a DataFrame as a heatmap.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (5, 5).

    Returns:
    - None: Displays a triangular heatmap of the correlation matrix with annotations.
    """

    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=size)

    # Mask to make it triangular
    mask = np.triu(np.ones_like(corr_matrix, dtype = np.bool_))
    sns.heatmap(corr_matrix, 
                annot=True, 
                vmin = -1, 
                vmax = 1, 
                mask=mask)
    

def plot_relation_tv_numeric(df, tv, size = (15, 10)):
    """
    Plots scatter plots showing the relationship between a target variable and all numeric columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - tv (str): The name of the target variable to analyze.
    - size (tuple, optional): The size of the entire plot grid as (width, height). Defaults to (15, 10).

    Returns:
    - None: Displays scatter plots of the target variable against each numeric column.
    """
    
    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = size)
    axes = axes.flat

    for i, col in enumerate(cols_num):

        if col == tv:
            fig.delaxes(axes[i])

        else:
            sns.scatterplot(x = col,
                        y = tv,
                        data = df_num,
                        ax = axes[i], 
                        palette = 'mako')
            
            axes[i].set_title(col)
            axes[i].set_xlabel('')

    # Remove last plot, if empty
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def plot_outliers(df, size=(9, 5)):
    """
    Plots boxplots for all numeric columns in the DataFrame to visualize outliers.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    - None: Displays a grid of boxplots for each numeric column, showing potential outliers.
    """

    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_rows = math.ceil(n_plots/2)

    cmap = plt.cm.get_cmap('mako', n_plots)
    color_list = [cmap(i) for i in range(cmap.N)]

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=size)
    axes = axes.flat

    for i, col in enumerate(cols_num):

        sns.boxplot(x = col, 
                    data = df_num,
                    ax = axes[i],
                    color=color_list[i]) 
        
        axes[i].set_title(f'{col} outliers')
        axes[i].set_xlabel('')

    # Remove last plot, if empty
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


def value_counts(df, col):
    """
    Calculates the value counts and proportions for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the column to analyze.

    Returns:
    - (pd.DataFrame): A DataFrame with two columns: the counts and proportions (rounded to two decimal places) of each unique value in the specified column.
    """

    print(f"The number of unique values for this category is {df[col].nunique()}")
    return pd.concat([df[col].value_counts(), df[col].value_counts(normalize=True).round(2)], axis=1)


def quick_plot_numeric(df, col, num=10, size=(10,5), rotation=45, color = 'mako'):
    """
    Generates a quick histogram plot for a numeric column, dividing the data into a specified number of bins.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col (str): The name of the numeric column to plot.
    - num (int, optional): The number of bins to divide the range into. Defaults to 10.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (10, 5).
    - rotation (int, optional): The rotation angle for x-axis labels. Defaults to 45.

    Returns:
    - None: Displays a histogram of the numeric column if the bin width is greater than or equal to 2.
    """

    max_ = df[col].max() + 1
    min_ = df[col].min()
    n = (max_ - min_) // num

    if n < 2:
        n = 1

    plot_numeric_distribution(df, min_, max_, col, n, size=size, rotation=rotation)


def plot_groupby(df, groupby, col, max_entries, size=(12, 6), palette='mako', method='median'):
    """
    Plots the median values of a numeric column grouped by a specified categorical column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - groupby (str): The name of the categorical column to group by.
    - col (str): The name of the numeric column for which medians are calculated.
    - max_entries (int): The maximum number of group entries to display.
    - size (tuple, optional): The size of the plot as (width, height). Defaults to (12, 6).
    - palette (str, optional): The color palette for the bar plot. Defaults to 'mako'.

    Returns:
    - None: Displays a bar plot of the median values per group.
    """

    if method=='median':

        grouped_df = (
            df.groupby(groupby)[col]
            .median()
            .sort_values()
            .reset_index()
        )

    elif method=='mean':
            grouped_df = (
        df.groupby(groupby)[col]
        .mean()
        .sort_values()
        .reset_index()
    )

    plt.figure(figsize=size)
    sns.barplot(
        data=grouped_df.iloc[:max_entries],
        y=col,
        x=groupby,
        palette=palette,
        edgecolor="black"
    )

    plt.title(f'{col} median per {groupby}')
    plt.xlabel('')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()


def plot_relation_tv(df, tv, size=(40, 40), n_cols = 2):
    """
    Plots the relationship of each column in the DataFrame with a target variable using histograms and count plots.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be visualized.
    - tv (str): The name of the target variable column to analyze relationships with.
    - size (tuple, optional): The size of the overall figure. Default is (40, 40).
    - n_cols (int, optional): The number of columns in the subplot grid. Default is 2.

    Returns:
    - None: The function directly displays the plots.
    """

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=['O', 'category']).columns

    fig, axes = plt.subplots(math.ceil(len(df.columns) / n_cols), n_cols, figsize=size)

    axes = axes.flat

    for i, col in enumerate(df.columns):
        if col == tv:
            fig.delaxes(axes[i])

        elif col in num_cols:
            sns.histplot(x = col, 
                            hue = tv, 
                            data = df, 
                            ax = axes[i], 
                            palette = "magma", 
                            legend = True)
            
        elif col in cat_cols:
            sns.countplot(x = col, 
                            hue = tv, 
                            data = df, 
                            ax = axes[i], 
                            palette = "magma"
                            )

        axes[i].set_title(f"{col} vs {tv}")   

    plt.tight_layout()