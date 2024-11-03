import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt


def analyze_lead_time_insights(df: pd.DataFrame) -> Dict:
    """
    Analyzes lead time insights from the provided DataFrame and calculates metrics related to the influence of lead time
    on cancellations, price, total nights, and booking conversion rates.

    Args:
        df (pd.DataFrame): The booking data DataFrame.

    Returns:
        dict: A dictionary with calculated lead time metrics and related insights.
    """
    # Remove high outliers for lead time analysis
    lead_time_cutoff = np.percentile(df['lead_time'], 95)
    lead_time_df = df[df['lead_time'] <= lead_time_cutoff].copy()

    # Lead time bins for grouping
    lead_time_bins = pd.cut(lead_time_df['lead_time'], bins=10)

    # Calculate metrics grouped by lead time bins
    cancellation_rate_by_lead_time = lead_time_df.groupby(lead_time_bins, observed=False)['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)
    avg_lead_time_with_children = lead_time_df.groupby('has_children', observed=False)['lead_time'].mean()
    avg_lead_time_by_total_guests = lead_time_df.groupby('total_guests', observed=False)['lead_time'].mean()
    avg_lead_time_by_month = lead_time_df.groupby('arrival_month', observed=False)['lead_time'].mean()
    avg_price_per_room_by_lead_time = lead_time_df.groupby(lead_time_bins, observed=False)['avg_price_per_room'].mean()
    avg_total_nights_by_lead_time = lead_time_df.groupby(lead_time_bins, observed=False)['total_nights'].mean()
    conversion_rate_by_lead_time = lead_time_df.groupby(lead_time_bins, observed=False)['booking_status'].apply(
        lambda x: (x == 'Not_Canceled').mean() * 100)

    # Percentile-based lead time analysis
    top_10_cutoff = np.percentile(df['lead_time'], 90)
    bottom_10_cutoff = np.percentile(df['lead_time'], 10)

    top_10_lead_time_df = df[df['lead_time'] >= top_10_cutoff].copy()
    bottom_10_lead_time_df = df[df['lead_time'] <= bottom_10_cutoff].copy()

    top_10_bins = pd.cut(top_10_lead_time_df['lead_time'], bins=5)
    bottom_10_bins = pd.cut(bottom_10_lead_time_df['lead_time'], bins=5)

    top_10_cancellation_rate = top_10_lead_time_df.groupby(top_10_bins, observed=False)['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)
    bottom_10_cancellation_rate = bottom_10_lead_time_df.groupby(bottom_10_bins, observed=False)[
        'booking_status'].apply(lambda x: (x == 'Canceled').mean() * 100)

    # Compile results into a dictionary
    lead_time_insights = {
        "Cancellation Rate by Lead Time": cancellation_rate_by_lead_time,
        "Average Lead Time with Children": avg_lead_time_with_children,
        "Average Lead Time by Total Guests": avg_lead_time_by_total_guests,
        "Average Lead Time by Month": avg_lead_time_by_month,
        "Average Price per Room by Lead Time": avg_price_per_room_by_lead_time,
        "Average Total Nights by Lead Time": avg_total_nights_by_lead_time,
        "Conversion Rate by Lead Time": conversion_rate_by_lead_time,
        "Top 10% Lead Time Cancellation Rate": top_10_cancellation_rate,
        "Bottom 10% Lead Time Cancellation Rate": bottom_10_cancellation_rate
    }

    return lead_time_insights


def plot_lead_time_analysis(df, save_fig_as=None):
    """
    Analyzes lead time insights from the provided DataFrame and generates visualizations
    related to lead time influence on booking cancellations and related metrics.

    Args:
        df (pd.DataFrame): The booking data DataFrame.
        save_fig_as (str, optional): The file path to save the plot as a PNG image. Defaults to None.
    """
    # Analyze lead time insights using the provided analysis function
    lead_time_insights = analyze_lead_time_insights(df)

    # Set up the main figure and subplots for lead time metrics
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle('Lead Time Analysis and Related Metrics', fontsize=20)

    # Plot each metric as defined in the helper code
    # 1. Cancellation Rate by Lead Time
    axes[0, 0].bar(lead_time_insights["Cancellation Rate by Lead Time"].index.astype(str),
                   lead_time_insights["Cancellation Rate by Lead Time"].values, color='salmon')
    axes[0, 0].set_title('Cancellation Rate by Lead Time')
    axes[0, 0].set_xlabel('Lead Time Range')
    axes[0, 0].set_ylabel('Cancellation Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Average Lead Time with Children
    axes[0, 1].bar(['Without Children', 'With Children'],
                   lead_time_insights["Average Lead Time with Children"].values, color=['skyblue', 'orange'])
    axes[0, 1].set_title('Average Lead Time by Presence of Children')
    axes[0, 1].set_ylabel('Average Lead Time (Days)')

    # 3. Average Lead Time by Total Guests
    axes[1, 0].plot(lead_time_insights["Average Lead Time by Total Guests"].index,
                    lead_time_insights["Average Lead Time by Total Guests"].values, marker='o', color='teal')
    axes[1, 0].set_title('Average Lead Time by Total Guests')
    axes[1, 0].set_xlabel('Total Guests')
    axes[1, 0].set_ylabel('Average Lead Time (Days)')

    # 4. Average Lead Time by Month
    axes[1, 1].bar(lead_time_insights["Average Lead Time by Month"].index,
                   lead_time_insights["Average Lead Time by Month"].values, color='purple')
    axes[1, 1].set_title('Average Lead Time by Month')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Lead Time (Days)')

    # 5. Average Price per Room by Lead Time
    axes[2, 0].plot(lead_time_insights["Average Price per Room by Lead Time"].index.astype(str),
                    lead_time_insights["Average Price per Room by Lead Time"].values, marker='o', color='blue')
    axes[2, 0].set_title('Average Price per Room by Lead Time')
    axes[2, 0].set_xlabel('Lead Time Range')
    axes[2, 0].set_ylabel('Average Price per Room')
    axes[2, 0].tick_params(axis='x', rotation=45)

    # 7. Average Total Nights by Lead Time
    axes[3, 0].plot(lead_time_insights["Average Total Nights by Lead Time"].index.astype(str),
                    lead_time_insights["Average Total Nights by Lead Time"].values, marker='o', color='navy')
    axes[3, 0].set_title('Average Total Nights by Lead Time')
    axes[3, 0].set_xlabel('Lead Time Range')
    axes[3, 0].set_ylabel('Average Total Nights')
    axes[3, 0].tick_params(axis='x', rotation=45)

    # 8. Conversion Rate by Lead Time
    axes[3, 1].plot(lead_time_insights["Conversion Rate by Lead Time"].index.astype(str),
                    lead_time_insights["Conversion Rate by Lead Time"].values, marker='o', color='green')
    axes[3, 1].set_title('Conversion Rate by Lead Time')
    axes[3, 1].set_xlabel('Lead Time Range')
    axes[3, 1].set_ylabel('Conversion Rate (%)')
    axes[3, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save main figure if requested
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()

    # Additional plots for top and bottom 10% lead times
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cancellation Rate for Top and Bottom 10% of Lead Time', fontsize=16)

    # Plot for top 10% lead times
    axes[0].bar(lead_time_insights["Top 10% Lead Time Cancellation Rate"].index.astype(str),
                lead_time_insights["Top 10% Lead Time Cancellation Rate"].values, color='crimson')
    axes[0].set_title('Cancellation Rate in Top 10% Lead Times')
    axes[0].set_xlabel('Lead Time Range (Top 10%)')
    axes[0].set_ylabel('Cancellation Rate (%)')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot for bottom 10% lead times
    axes[1].bar(lead_time_insights["Bottom 10% Lead Time Cancellation Rate"].index.astype(str),
                lead_time_insights["Bottom 10% Lead Time Cancellation Rate"].values, color='teal')
    axes[1].set_title('Cancellation Rate in Bottom 10% Lead Times')
    axes[1].set_xlabel('Lead Time Range (Bottom 10%)')
    axes[1].set_ylabel('Cancellation Rate (%)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save additional figure if requested
    if save_fig_as:
        plt.savefig(save_fig_as.replace('.png', '_additional.png'), format='png')

    plt.show()
