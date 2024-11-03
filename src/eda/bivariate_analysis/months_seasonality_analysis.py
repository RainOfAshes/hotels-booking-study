import matplotlib.pyplot as plt
import pandas as pd


def analyze_monthly_cancellation_rate(df: pd.DataFrame):
    """
    Analyzes monthly cancellation rate along with related metrics for average calculations (using all data)
    and summation/total-based calculations (using only 2018 data).

    Args:
        df (pd.DataFrame): The booking data DataFrame.

    Returns:
        dict: A dictionary with calculated metrics for monthly cancellation rates and related insights.
    """
    # Filter for 2018 data only for sum and volume-based calculations
    df_2018 = df[df['arrival_year'] == 2018].copy()

    # Monthly cancellation rate (2018 only)
    cancellation_rate_by_month = df_2018.groupby('arrival_month')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Monthly average price per room and per person (2018 only)
    avg_price_per_room_by_month = df_2018.groupby('arrival_month')['avg_price_per_room'].mean()

    # Monthly booking volume percentage (2018 only)
    monthly_booking_volume = df_2018['arrival_month'].value_counts(normalize=True).sort_index() * 100

    # Average total nights, lead time, and special requests by month (all data)
    avg_total_nights_by_month = df.groupby('arrival_month')['total_nights'].mean()
    avg_lead_time_by_month = df.groupby('arrival_month')['lead_time'].mean()
    avg_special_requests_by_month = df.groupby('arrival_month')['no_of_special_requests'].mean()

    # Monthly cancellations count (2018 only)
    monthly_cancellations = df_2018[df_2018['booking_status'] == 'Canceled'].groupby('arrival_month').size()

    # Compile results into a dictionary
    monthly_analysis_results = {
        "Cancellation Rate by Month (%)": cancellation_rate_by_month,
        "Avg Price per Room by Month": avg_price_per_room_by_month,
        "Monthly Booking Volume (%)": monthly_booking_volume,
        "Avg Total Nights by Month": avg_total_nights_by_month,
        "Avg Lead Time by Month": avg_lead_time_by_month,
        "Avg Special Requests by Month": avg_special_requests_by_month,
        "Monthly Cancellations (Count)": monthly_cancellations
    }

    return monthly_analysis_results


def plot_monthly_analysis(df: pd.DataFrame, save_fig_as=None):
    """
    Plots the monthly analysis of key metrics for the arrival year 2018.

    Args:
        df (pd.DataFrame): data frame containing the booking data.
        save_fig_as (str, optional): File path to save the plot as a PNG image. Defaults to None.
    """
    arrival_month_analysis_results = analyze_monthly_cancellation_rate(df)
    fig, axes = plt.subplots(3, 2, figsize=(18, 24))
    fig.suptitle('Monthly Analysis of Key Metrics for Arrival Year 2018', fontsize=16)

    # 1. Cancellation Rate by Month (Line Chart)
    axes[0, 0].plot(arrival_month_analysis_results["Cancellation Rate by Month (%)"].index,
                    arrival_month_analysis_results["Cancellation Rate by Month (%)"].values, marker='o', color='crimson')
    axes[0, 0].set_title('Cancellation Rate by Month')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Cancellation Rate (%)')

    # 2. Average Price per Room by Month (Line Chart)
    axes[0, 1].plot(arrival_month_analysis_results["Avg Price per Room by Month"].index,
                    arrival_month_analysis_results["Avg Price per Room by Month"].values, marker='o', color='blue')
    axes[0, 1].set_title('Average Price per Room by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Avg Price per Room')

    # 4. Booking Volume by Month (Line Chart)
    axes[1, 1].plot(arrival_month_analysis_results["Monthly Booking Volume (%)"].index,
                    arrival_month_analysis_results["Monthly Booking Volume (%)"].values, marker='o', color='purple')
    axes[1, 1].set_title('Booking Volume by Month')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Percentage of Total Bookings')

    # 5. Average Total Nights by Month (Line Chart)
    axes[2, 0].plot(arrival_month_analysis_results["Avg Total Nights by Month"].index,
                    arrival_month_analysis_results["Avg Total Nights by Month"].values, marker='o', color='orange')
    axes[2, 0].set_title('Average Total Nights by Month')
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Average Total Nights')

    # 6. Average Lead Time by Month (Line Chart)
    axes[2, 1].plot(arrival_month_analysis_results["Avg Lead Time by Month"].index,
                    arrival_month_analysis_results["Avg Lead Time by Month"].values, marker='o', color='teal')
    axes[2, 1].set_title('Average Lead Time by Month')
    axes[2, 1].set_xlabel('Month')
    axes[2, 1].set_ylabel('Average Lead Time (Days)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure if requested
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()
