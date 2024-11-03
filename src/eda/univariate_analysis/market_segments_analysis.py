import pandas as pd
import matplotlib.pyplot as plt


def marketing_segments_insights_generation(df: pd.DataFrame):
    """
    Generates insights for marketing segments from the provided DataFrame.
    """
    insights = {
        "percentage_by_segment": df['market_segment_type'].value_counts(normalize=True)[:3] * 100,
        # Only top 3 segments
        "avg_total_nights_by_segment": df.groupby('market_segment_type')['total_nights'].mean(),
        "avg_lead_time_by_segment": df.groupby('market_segment_type')['lead_time'].mean(),
        "cancellation_rate_by_segment": df.groupby('market_segment_type')['booking_status'].apply(
            lambda x: (x == 'Canceled').mean() * 100),
        "special_requests_by_segment": df.groupby('market_segment_type')['no_of_special_requests'].mean(),
        "avg_weekend_nights_by_segment": df.groupby('market_segment_type')['no_of_weekend_nights'].mean(),
        "avg_week_nights_by_segment": df.groupby('market_segment_type')['no_of_week_nights'].mean()
    }
    return insights


def plot_marketing_segments_insights(df, save_fig_as=None):
    """
    Generates plots for marketing segments insights and optionally saves the figure as a PNG image.

    Args:
        df (pd.DataFrame): The input DataFrame containing booking data.
        save_fig_as (str, optional): The file path to save the plot as a PNG image. Defaults to None.
    """
    # Generate insights dictionary using the first function
    insights = marketing_segments_insights_generation(df)

    # Setting up the layout for multiple subplots
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle('Market Segment Analysis', fontsize=18)

    # 1. Percentage of Bookings by Top 3 Segments (Pie Chart)
    axes[0, 0].pie(insights["percentage_by_segment"].values, labels=insights["percentage_by_segment"].index,
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Percentage of Bookings by Top 3 Segments')

    # 2. Average Total Nights by Segment (Bar Chart) - Exclude any empty entries in the plot
    total_nights_index = insights["avg_total_nights_by_segment"].index
    total_nights_values = insights["avg_total_nights_by_segment"].values
    axes[1, 0].bar(total_nights_index, total_nights_values, color='orange')
    axes[1, 0].set_title('Average Total Nights by Segment')
    axes[1, 0].set_xlabel('Market Segment')
    axes[1, 0].set_ylabel('Avg Total Nights')

    # 3. Lead Time by Segment (Bar Chart)
    axes[1, 1].bar(insights["avg_lead_time_by_segment"].index, insights["avg_lead_time_by_segment"].values,
                   color='teal')
    axes[1, 1].set_title('Lead Time by Segment')
    axes[1, 1].set_xlabel('Market Segment')
    axes[1, 1].set_ylabel('Avg Lead Time (Days)')

    # 4. Cancellation Rate by Segment (Bar Chart)
    axes[2, 0].bar(insights["cancellation_rate_by_segment"].index, insights["cancellation_rate_by_segment"].values,
                   color='crimson')
    axes[2, 0].set_title('Cancellation Rate by Segment')
    axes[2, 0].set_xlabel('Market Segment')
    axes[2, 0].set_ylabel('Cancellation Rate (%)')

    # 5. Special Requests by Segment (Bar Chart)
    axes[2, 1].bar(insights["special_requests_by_segment"].index, insights["special_requests_by_segment"].values,
                   color='green')
    axes[2, 1].set_title('Avg Special Requests by Segment')
    axes[2, 1].set_xlabel('Market Segment')
    axes[2, 1].set_ylabel('Average Special Requests')

    # 6. Average Weekend Nights by Segment (Bar Chart)
    axes[3, 0].bar(insights["avg_weekend_nights_by_segment"].index, insights["avg_weekend_nights_by_segment"].values,
                   color='skyblue')
    axes[3, 0].set_title('Avg Weekend Nights by Segment')
    axes[3, 0].set_xlabel('Market Segment')
    axes[3, 0].set_ylabel('Avg Weekend Nights')

    # 7. Average Week Nights by Segment (Bar Chart)
    axes[3, 1].bar(insights["avg_week_nights_by_segment"].index, insights["avg_week_nights_by_segment"].values,
                   color='darkblue')
    axes[3, 1].set_title('Avg Week Nights by Segment')
    axes[3, 1].set_xlabel('Market Segment')
    axes[3, 1].set_ylabel('Avg Week Nights')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title space

    # Save figure if a path is provided
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()
