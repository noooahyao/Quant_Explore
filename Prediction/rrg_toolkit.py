import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import numpy as np
from typing import List

def calculate_rs_ratios(prices, benchmark):
    """
    Calculate the RS ratios relative to a benchmark.
    Parameters:
    prices (pd.DataFrame): A DataFrame containing the ETF prices.
    benchmark (str): The ticker symbol of the benchmark.
    Returns:
    pd.DataFrame: A DataFrame containing the RS ratios.
    """
    benchmark_prices = prices[benchmark]
    rs_ratios = pd.DataFrame()
    for etf in prices.columns:
        rs_ratio = prices[etf] / benchmark_prices
        rs_ratios[etf] = rs_ratio

    # rs_ratios.columns = pd.MultiIndex.from_product([rs_ratios.columns, ['RS Ratio']])

    return rs_ratios

def calc_jdk_rs_ratio(data: pd.Series) -> pd.Series:
    """
    Calculate the JDK RS Ratio.
    Parameters:
    data (pd.Series): A pandas series containing the data to calculate the ratio from.
    Returns:
    pd.Series: A pandas series containing the JDK RS Ratio.
    """
    # Calculate the 50-day and 200-day moving averages
    ma_50 = data.rolling(window=50).mean()
    ma_200 = data.rolling(window=200).mean()

    # Calculate the JDK RS Ratio
    jdk_rs_ratio = ma_50 / ma_200 - 1

    return jdk_rs_ratio


def calc_jdk_rs_momentum(jdk_rs_ratio: pd.Series) -> pd.Series:
    """
    Calculate the JDK RS Momentum.
    Parameters:
    jdk_rs_ratio (pd.Series): A pandas series containing the JDK RS Ratio to calculate the momentum from.
    Returns:
    pd.Series: A pandas series containing the JDK RS Momentum.
    """
    # Calculate the 20-day momentum
    jdk_rs_momentum = jdk_rs_ratio.diff(20)

    return jdk_rs_momentum


def calculate_jdk_rs(data: pd.Series) -> pd.Series:
    """
    Calculate the JDK RS Ratio and Momentum.
    Parameters:
    data (pd.Series): A pandas series containing the data to calculate the ratio and momentum from.
    Returns:
    pd.Series: A pandas series containing the JDK RS Ratio and Momentum.
    """
    if len(data) < 200:
        raise ValueError("Input data must have at least 200 data points")

    # Calculate the JDK RS Ratio
    jdk_rs_ratio = calc_jdk_rs_ratio(data)

    # Calculate the JDK RS Momentum
    jdk_rs_momentum = calc_jdk_rs_momentum(jdk_rs_ratio)

    # Concatenate the JDK RS Ratio and Momentum into a single series
    jdk_rs = pd.concat([jdk_rs_ratio, jdk_rs_momentum], axis=1)
    jdk_rs.columns = ['JDK RS Ratio', 'JDK RS Momentum']

    return jdk_rs

def categorize_point(x, y):
    """
    Categorizes a point based on its coordinates (x, y) into one of four categories.
   
    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        
    Returns:
        str: The category of the point.
            - 'Leading' if the point is in the top right quadrant (x >= 0 and y >= 0).
            - 'Improving' if the point is in the top left quadrant (x < 0 and y >= 0).
            - 'Weakening' if the point is in the bottom right quadrant (x >= 0 and y < 0).
            - 'Lagging' if the point is in the bottom left quadrant (x < 0 and y < 0).
    """
    if x >= 0 and y >= 0:
        return 'Leading'
    elif x < 0 and y >= 0:
        return 'Improving'
    elif x >= 0 and y < 0:
        return 'Weakening'
    else:
        return 'Lagging'

def categorize_radius(x, y, circle_radius):
    """
    Categorizes a signal based on its coordinates (x, y) relative to a given circle radius.
    
    Args:
        x (float): The x-coordinate of the signal.
        y (float): The y-coordinate of the signal.
        circle_radius (float): The radius of the circle used for categorization.
        
    Returns:
        str: The category of the signal.
            - 'Weak' if the signal is within or on the circle boundary (distance <= circle_radius).
            - 'Strong' if the signal is outside the circle boundary (distance > circle_radius).
    """
    distance = np.sqrt(x**2 + y**2)
    if distance <= circle_radius:
        return 'Weak'
    else:
        return 'Strong'


def calculate_slope(x1, y1, x2, y2):
    """
    Calculate the slope between two points.
    Arguments:
    x1, y1: The coordinates of the first point.
    x2, y2: The coordinates of the second point.
    Returns:
    The slope between the two points.
    """
    slope = (y2 - y1) / (x2 - x1)
    return slope



def extrapolate_coordinates(x_series, y_series, j):
    """
    Extrapolates coordinates based on historical and current values and calculates the projected category.

    Args:
        x_series (pandas.Series): Series containing x-coordinates.
        y_series (pandas.Series): Series containing y-coordinates.
        j (int): Index of the current coordinate in the series.

    Returns:
        tuple: A tuple containing the following values:
            - x_point (float): Current x-coordinate.
            - y_point (float): Current y-coordinate.
            - projected_x (float): Projected x-coordinate.
            - projected_y (float): Projected y-coordinate.
            - moving_towards (str): Categorized direction of movement.
            - x_historic (float): Historic x-coordinate.
            - y_historic (float): Historic y-coordinate.
            - projected_category (str): Categorized direction of projected movement.
    """
    x_point = x_series[j]
    y_point = y_series[j]

    projected_x = x_series[j+1]
    projected_y = y_series[j+1]
    moving_towards = categorize_point(projected_x, projected_y)

    x_historic = x_series[j-5]
    y_historic = y_series[j-5]

    angle_historic_current = np.arctan2(y_point - y_historic, x_point - x_historic)

    length_historic_current = np.sqrt((x_point - x_historic)**2 + (y_point - y_historic)**2)
    length_historic_projected = length_historic_current
    length_current_projected = length_historic_current

    projected_x = x_point + length_current_projected * np.cos(angle_historic_current)
    projected_y = y_point + length_current_projected * np.sin(angle_historic_current)

    x_change = projected_x - x_point
    y_change = projected_y - y_point
    projected_category = categorize_point(x_change, y_change)

    return x_point, y_point, projected_x, projected_y, moving_towards, x_historic, y_historic, projected_category



def extract_coordinates(etf_names, jdk_rs_ratios, jdk_rs_momentums, rs_ratios, trailing_days=20, circle_radius=0.1):
    """
    Extracts coordinates and related information for a given set of ETFs.
    
    Args:
        etf_names (list): List of ETF names.
        jdk_rs_ratios (pandas.DataFrame): DataFrame containing JDK RS ratios.
        jdk_rs_momentums (pandas.DataFrame): DataFrame containing JDK RS momentums.
        rs_ratios (pandas.DataFrame): DataFrame containing RS ratios.
        trailing_days (int): Number of trailing days to consider for extrapolation (default: 20).
        circle_radius (float): Circle radius for signal categorization (default: 0.1).
        
    Returns:
        pandas.DataFrame: DataFrame containing the extracted coordinates and information.
    """
    coordinates = pd.DataFrame(columns=['ETF', 'Date', 'Current X', 'Current Y', 'Current Category', 'Current Signal Strength',
                                        'Historic X', 'Historic Y', 'Angle between Historic and Current', 'Projected X',
                                        'Projected Y', 'Projected Category', 'Projected Signal Strength'])

    for etf in etf_names:
        x_series = jdk_rs_ratios[(etf, 'RS Ratio')]
        y_series = jdk_rs_momentums[(etf, 'RS Ratio')]
        dates = x_series.index

        for j in range(trailing_days, len(x_series)-1):
            x_point, y_point, projected_x, projected_y, moving_towards, x_historic, y_historic, projected_category = extrapolate_coordinates(x_series, y_series, j)

            category = categorize_point(x_point, y_point)
            signal = categorize_radius(x_point, y_point, circle_radius)
            historic_x = x_historic
            historic_y = y_historic

            coordinates = coordinates.append({
                'ETF': etf,
                'Date': dates[j],
                'Current X': x_point,
                'Current Y': y_point,
                'Current Category': category,
                'Current Signal Strength': signal,
                'Historic X': historic_x,
                'Historic Y': historic_y,
                'Angle between Historic and Current': np.arctan2(y_point - y_historic, x_point - x_historic),
                'Projected X': projected_x,
                'Projected Y': projected_y,
                'Projected Category': moving_towards,
                'Projected Signal Strength': signal
            }, ignore_index=True)

    return coordinates

# Default ETF names
def visualize_rrg(jdk_rs_ratios, jdk_rs_momentums, rs_ratios,etf_names= ['SPY', 'SPY', 'SPY', 'SPY'] , trailing_days=20):
    
    """
    Visualize the RRG for the given ETF names.
    Parameters:
    etf_names (list): A list of ETF names to be visualized.
    jdk_rs_ratios (pd.DataFrame): DataFrame containing JDK RS Ratios.
    jdk_rs_momentums (pd.DataFrame): DataFrame containing JDK RS Momentums.
    rs_ratios (pd.DataFrame): DataFrame containing RS Ratios.
    trailing_days (int): Number of days in the trailing line.
    """ 
  
    # Create a figure and axis object
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    # Set the axis limits for all subplots
    for ax in axs.flat:
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)

             
    # Loop through each ETF and plot its data in a separate subplot
    for i, etf in enumerate(etf_names):
        x, y = jdk_rs_ratios[(etf, 'RS Ratio')].iloc[-1], jdk_rs_momentums[(etf, 'RS Ratio')].iloc[-1]
        ax = axs[i // 2, i % 2]
        ax.scatter(x, y, color='red', alpha=1.00, linewidth=2)  # use a single color for all the plots

        # Create a dataframe for the last n days of data
        trailing_data = rs_ratios.iloc[-trailing_days:]

        # Plot the trailing line for each ETF
        x_trailing = jdk_rs_ratios[(etf, 'RS Ratio')].loc[trailing_data.index]
        y_trailing = jdk_rs_momentums[(etf, 'RS Ratio')].loc[trailing_data.index]
        ax.plot(x_trailing, y_trailing, color='red', alpha=0.50, linewidth=2)

        # Add dots on the trailing line for every 14 days
        for j in range(0, len(x_trailing), 14):
            ax.scatter(x_trailing.iloc[j], y_trailing.iloc[j], color='black', alpha=0.3, s=10)
        
        # Define the colors for each quadrant
        colors2 = {"Leading": "darkgreen", "Lagging": "darkred", "Improving": "orange", "Weakening": "darkblue"}

        # Add rectangular patches to shade the quadrants
        ax.add_patch(mpl.patches.Rectangle((-1, -1), 1, 1, color=colors2["Lagging"], alpha=0.1))
        ax.add_patch(mpl.patches.Rectangle((0, -1), 1, 1, color=colors2["Improving"], alpha=0.1))
        ax.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, color=colors2["Leading"], alpha=0.1))
        ax.add_patch(mpl.patches.Rectangle((-1, 0), 1, 1, color=colors2["Weakening"], alpha=0.1))

        # Add a circle at the center
        center = (0, 0)  # center coordinates
        radius = 0.1  # radius of the circle
        circle = plt.Circle(center, radius, facecolor='none', edgecolor='gray', alpha=0.3, linewidth=2)
        ax.add_patch(circle)
                
        # Add x-axis and y-axis lines
        ax.axhline(y=0, color='gray')
        ax.axvline(x=0, color='gray')

        # Add labels for the quadrants
        ax.text(-0.15, 0.15, "Improving", fontsize=8, ha="center", va="center")
        ax.text(-0.15, -0.15, "Lagging", fontsize=8, ha="center", va="center")
        ax.text(0.15, -0.15, "Weakening", fontsize=8, ha="center", va="center")
        ax.text(0.15, 0.15, "Leading", fontsize=8, ha="center", va="center")

        # Add legend
        ax.legend([etf], loc="upper center", fontsize=8)


        
# Set the titles for each subplot
        for i, etf in enumerate(etf_names):
            ax = axs[i // 2, i % 2]
            ax.set_title(etf)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

