import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def calculate_cancellation_metrics(df):
    """
    Calculates various cancellation metrics from the booking DataFrame.

    Args:
        df (pd.DataFrame): The booking data DataFrame.

    Returns:
        dict: A dictionary with calculated cancellation metrics.
    """
    # Basic Cancellation Metrics
    total_bookings = len(df)
    cancellations = df['booking_status'].value_counts().get('Canceled', 0)
    cancellation_rate = (cancellations / total_bookings) * 100

    # Filter data for 2018 to focus on recent trends
    df_2018 = df[df['arrival_year'] == 2018]

    # Monthly Cancellation Rate for 2018
    monthly_cancellations = df_2018[df_2018['booking_status'] == 'Canceled'].groupby('arrival_month').size()
    total_bookings_per_month = df_2018.groupby('arrival_month').size()
    monthly_cancellation_rate = (monthly_cancellations / total_bookings_per_month) * 100

    # Lead Time Comparison
    lead_time_cancelled = df[df['booking_status'] == 'Canceled']['lead_time'].mean()
    lead_time_not_cancelled = df[df['booking_status'] != 'Canceled']['lead_time'].mean()

    # Special Requests Comparison
    special_requests_cancelled = df[df['booking_status'] == 'Canceled']['no_of_special_requests'].mean()
    special_requests_not_cancelled = df[df['booking_status'] != 'Canceled']['no_of_special_requests'].mean()

    # Cancellation Rate Over Time (for 2018)
    cancellations_over_time = df_2018[df_2018['booking_status'] == 'Canceled'].groupby(
        ['arrival_year', 'arrival_month']).size()
    total_bookings_over_time = df_2018.groupby(['arrival_year', 'arrival_month']).size()
    cancellation_rate_over_time = (cancellations_over_time / total_bookings_over_time) * 100

    # Average Price Comparison
    avg_price_cancelled = df[df['booking_status'] == 'Canceled']['avg_price_per_room'].mean()
    avg_price_not_cancelled = df[df['booking_status'] != 'Canceled']['avg_price_per_room'].mean()

    # Additional Metrics
    percent_with_children = (df['has_children'].mean()) * 100

    # Cancellation by Children Presence
    cancellation_with_children = df.groupby('has_children')['booking_status'].value_counts(
        normalize=True).unstack().fillna(0)

    # Cancellation Rates by Week Nights and Weekend Nights
    cancellation_rate_by_week_nights = df.groupby('no_of_week_nights')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)
    cancellation_rate_by_weekend_nights = df.groupby('no_of_weekend_nights')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Cancellation Rate by Market Segment
    cancellation_rate_by_segment = df.groupby('market_segment_type')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Cancellation Rate by Repeated Guest Status
    cancellation_rate_repeated = df.groupby('repeated_guest')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Cancellation Rate by Lead Time Bins
    lead_time_bins = pd.cut(df['lead_time'], bins=10)
    cancellation_rate_by_lead_time = df.groupby(lead_time_bins, observed=False)['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Return metrics as a dictionary
    return {
        "Overall Cancellation Rate (%)": cancellation_rate,
        "Monthly Cancellation Rate (%)": monthly_cancellation_rate,
        "Lead Time - Cancelled (Days)": lead_time_cancelled,
        "Lead Time - Not Cancelled (Days)": lead_time_not_cancelled,
        "Avg Special Requests - Cancelled": special_requests_cancelled,
        "Avg Special Requests - Not Cancelled": special_requests_not_cancelled,
        "Cancellation Rate Over Time (%)": cancellation_rate_over_time,
        "Avg Price - Cancelled Bookings": avg_price_cancelled,
        "Avg Price - Not Cancelled Bookings": avg_price_not_cancelled,
        "Cancellation Rate by Segment (%)": cancellation_rate_by_segment,
        "Cancellation Rate by Repeated Status (%)": cancellation_rate_repeated,
        "Cancellation Rate by Lead Time (%)": cancellation_rate_by_lead_time,
        "Cancellation Rate by Week Nights (%)": cancellation_rate_by_week_nights,
        "Cancellation Rate by Weekend Nights (%)": cancellation_rate_by_weekend_nights,
        "Percent with Children": percent_with_children,
        "Cancellation with Children": cancellation_with_children.reset_index()
    }


def plot_cancellation_metrics(df, save_fig_as=None):
    """
    Generates comprehensive plots for cancellation metrics from the booking DataFrame
    and optionally saves the figure as a PNG image.

    Args:
        df (pd.DataFrame): The booking data DataFrame.
        save_fig_as (str, optional): The file path to save the plot as a PNG image. Defaults to None.
    """
    # Calculate cancellation metrics using the existing function
    metrics = calculate_cancellation_metrics(df)

    # Create figure and axes for multiple subplots
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle('Comprehensive Analysis of Booking Cancellations', fontsize=16)

    # 1. Overall Booking Status Count Plot
    sns.countplot(data=df, x='booking_status', ax=axes[0, 0])
    axes[0, 0].set_title('Overall Booking Status')
    axes[0, 0].set_xlabel('Booking Status')
    axes[0, 0].set_ylabel('Count')

    # 2. Monthly Cancellation Rate Bar Plot
    metrics['Monthly Cancellation Rate (%)'].plot(kind='bar', color='skyblue', ax=axes[0, 1])
    axes[0, 1].set_title('Monthly Cancellation Rate (%)')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Cancellation Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Lead Time Analysis for Cancellations (Histogram)
    axes[1, 0].hist(df.loc[df['booking_status'] == 'Canceled', 'lead_time'].dropna(), bins=30, alpha=0.5,
                    label='Canceled')
    axes[1, 0].hist(df.loc[df['booking_status'] != 'Canceled', 'lead_time'].dropna(), bins=30, alpha=0.5,
                    label='Not Canceled')
    axes[1, 0].set_title('Lead Time Distribution by Booking Status')
    axes[1, 0].set_xlabel('Lead Time (Days)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # 4. Influence of Special Requests on Cancellations (Histogram)
    axes[1, 1].hist(df.loc[df['booking_status'] == 'Canceled', 'no_of_special_requests'].dropna(), bins=6, alpha=0.5,
                    label='Canceled')
    axes[1, 1].hist(df.loc[df['booking_status'] != 'Canceled', 'no_of_special_requests'].dropna(), bins=6, alpha=0.5,
                    label='Not Canceled')
    axes[1, 1].set_title('Distribution of Special Requests by Booking Status')
    axes[1, 1].set_xlabel('Number of Special Requests')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    # 5. Cancellation Rate by Lead Time (Bar Chart)
    metrics['Cancellation Rate by Lead Time (%)'].plot(kind='bar', color='salmon', ax=axes[2, 0])
    axes[2, 0].set_title('Cancellation Rate by Lead Time')
    axes[2, 0].set_xlabel('Lead Time Range')
    axes[2, 0].set_ylabel('Cancellation Rate (%)')
    axes[2, 0].tick_params(axis='x', rotation=45)

    # 6. Cancellation Rate by Segment (Bar Chart)
    metrics['Cancellation Rate by Segment (%)'].plot(kind='bar', color='crimson', ax=axes[2, 1])
    axes[2, 1].set_title('Cancellation Rate by Segment')
    axes[2, 1].set_xlabel('Market Segment')
    axes[2, 1].set_ylabel('Cancellation Rate (%)')

    # 7. Cancellation Rate by Repeated Status (Bar Chart)
    metrics['Cancellation Rate by Repeated Status (%)'].plot(kind='bar', color=['green', 'purple'], ax=axes[3, 0])
    axes[3, 0].set_title('Cancellation Rate by Repeated Status')
    axes[3, 0].set_xlabel('Repeated Guest Status')
    axes[3, 0].set_ylabel('Cancellation Rate (%)')
    axes[3, 0].set_xticklabels(['Non-Repeated', 'Repeated'])

    # 8. Room Price Distribution by Booking Status (Box Plot)
    sns.boxplot(data=df, x='booking_status', y='avg_price_per_room', ax=axes[3, 1])
    axes[3, 1].set_title('Room Price Distribution by Booking Status')
    axes[3, 1].set_xlabel('Booking Status')
    axes[3, 1].set_ylabel('Average Price per Room')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to fit title and labels

    # Save figure if a path is provided
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()
