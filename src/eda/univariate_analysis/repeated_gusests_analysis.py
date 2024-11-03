import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_returning_customers(df):
    """
    Analyzes the behavior of returning customers in the booking data and calculates various metrics
    related to repeated guests.

    Args:
        df (pd.DataFrame): The booking data DataFrame.

    Returns:
        dict: A dictionary with calculated metrics for returning customers.
    """
    # Percentage of returning customers
    total_customers = len(df)
    returning_customers = df['repeated_guest'].sum()
    returning_percentage = (returning_customers / total_customers) * 100

    # Monthly booking percentage for repeated guests
    monthly_booking_repeated_guests = df[df['repeated_guest'] == 1].groupby('arrival_month').size() / df.groupby(
        'arrival_month').size() * 100

    # Average special requests by repeated guest status
    avg_special_requests_repeated = df.groupby('repeated_guest')['no_of_special_requests'].mean()

    # Cancellation rate for repeated vs. non-repeated guests
    cancellation_rate_repeated = df.groupby('repeated_guest')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Average total guests by repeated guest status
    avg_total_guests_repeated = df.groupby('repeated_guest')['total_guests'].mean()

    # Meal preference distribution for repeated guests
    meal_preference_repeated = df[df['repeated_guest'] == 1]['type_of_meal_plan'].value_counts(normalize=True) * 100

    # Percentage of repeated guests with children
    has_children_repeated = df.groupby('repeated_guest')['has_children'].mean() * 100  # Percentage with children

    # Average total nights by repeated guest status
    avg_total_nights_repeated = df.groupby('repeated_guest')['total_nights'].mean()

    # Correlation of previous reservations with other factors
    correlation_with_reservations = \
    df[['no_previous_reservations', 'total_nights', 'lead_time', 'no_of_special_requests']].corr()[
        'no_previous_reservations']

    # Average metrics by previous reservations
    avg_metrics_by_previous_reservations = df.groupby('no_previous_reservations')[
        ['total_nights', 'lead_time', 'no_of_special_requests']].mean()

    # Counts of repeated guests by market segment
    market_segment_repeated_counts = df.groupby(['market_segment_type', 'repeated_guest']).size().unstack().fillna(0)
    market_segment_repeated_percentage = market_segment_repeated_counts.div(market_segment_repeated_counts.sum(axis=1),
                                                                            axis=0) * 100

    # Compile results
    return {
        "Returning Customers (%)": returning_percentage,
        "Monthly Booking - Repeated Guests (%)": monthly_booking_repeated_guests,
        "Avg Special Requests - Repeated vs Non-Repeated": avg_special_requests_repeated,
        "Cancellation Rate - Repeated vs Non-Repeated (%)": cancellation_rate_repeated,
        "Avg Total Guests - Repeated vs Non-Repeated": avg_total_guests_repeated,
        "Meal Preference - Repeated Guests (%)": meal_preference_repeated,
        "Has Children - Repeated vs Non-Repeated (%)": has_children_repeated,
        "Avg Total Nights - Repeated vs Non-Repeated": avg_total_nights_repeated,
        "Correlation with Previous Reservations": correlation_with_reservations,
        "Avg Metrics by Previous Reservations": avg_metrics_by_previous_reservations,
        "Market Segment - Repeated Guests (%)": market_segment_repeated_percentage
    }


def plot_returning_customer_analysis(df, save_fig_as=None):
    """
    Generates a comprehensive set of plots analyzing the behavior and metrics of returning customers
    and optionally saves the figure as a PNG image.

    Args:
        df (pd.DataFrame): The booking data DataFrame.
        save_fig_as (str, optional): The file path to save the plot as a PNG image. Defaults to None.
    """
    # Calculate metrics for returning customers using the existing function
    metrics = analyze_returning_customers(df)

    # Set up figure with multiple subplots
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle('Analysis of Returning Guest Behavior and Metrics', fontsize=20)

    # 1. Monthly Booking Patterns for Repeated Guests (Line Chart)
    axes[0, 0].plot(metrics["Monthly Booking - Repeated Guests (%)"].index,
                    metrics["Monthly Booking - Repeated Guests (%)"].values, marker='o', color='crimson')
    axes[0, 0].set_title('Monthly Booking Patterns for Repeated Guests')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Percentage of Bookings')

    # 2. Average Special Requests by Repeated Status (Bar Chart)
    axes[0, 1].bar(metrics["Avg Special Requests - Repeated vs Non-Repeated"].index,
                   metrics["Avg Special Requests - Repeated vs Non-Repeated"].values, color=['skyblue', 'orange'])
    axes[0, 1].set_title('Average Special Requests by Repeated Status')
    axes[0, 1].set_xlabel('Repeated Guest Status')
    axes[0, 1].set_ylabel('Average Special Requests')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['Non-Repeated', 'Repeated'])

    # 3. Cancellation Rate by Repeated Status (Bar Chart)
    axes[1, 0].bar(metrics["Cancellation Rate - Repeated vs Non-Repeated (%)"].index,
                   metrics["Cancellation Rate - Repeated vs Non-Repeated (%)"].values, color=['green', 'purple'])
    axes[1, 0].set_title('Cancellation Rate by Repeated Status')
    axes[1, 0].set_xlabel('Repeated Guest Status')
    axes[1, 0].set_ylabel('Cancellation Rate (%)')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['Non-Repeated', 'Repeated'])

    # 4. Average Total Guests by Repeated Status (Bar Chart)
    axes[1, 1].bar(metrics["Avg Total Guests - Repeated vs Non-Repeated"].index,
                   metrics["Avg Total Guests - Repeated vs Non-Repeated"].values, color=['steelblue', 'coral'])
    axes[1, 1].set_title('Average Total Guests by Repeated Status')
    axes[1, 1].set_xlabel('Repeated Guest Status')
    axes[1, 1].set_ylabel('Average Total Guests')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Non-Repeated', 'Repeated'])

    # 5. Meal Preference for Repeated Guests (Pie Chart)
    axes[2, 0].pie(metrics["Meal Preference - Repeated Guests (%)"].values,
                   labels=metrics["Meal Preference - Repeated Guests (%)"].index, autopct='%1.1f%%', startangle=140)
    axes[2, 0].set_title('Meal Preference for Repeated Guests')

    # 6. Has Children by Repeated Status (Bar Chart)
    axes[2, 1].bar(metrics["Has Children - Repeated vs Non-Repeated (%)"].index,
                   metrics["Has Children - Repeated vs Non-Repeated (%)"].values, color=['grey', 'gold'])
    axes[2, 1].set_title('Percentage with Children by Repeated Status')
    axes[2, 1].set_xlabel('Repeated Guest Status')
    axes[2, 1].set_ylabel('Percentage with Children')
    axes[2, 1].set_xticks([0, 1])
    axes[2, 1].set_xticklabels(['Non-Repeated', 'Repeated'])

    # 7. Average Total Nights by Repeated Status (Bar Chart)
    axes[3, 0].bar(metrics["Avg Total Nights - Repeated vs Non-Repeated"].index,
                   metrics["Avg Total Nights - Repeated vs Non-Repeated"].values, color=['navy', 'pink'])
    axes[3, 0].set_title('Average Total Nights by Repeated Status')
    axes[3, 0].set_xlabel('Repeated Guest Status')
    axes[3, 0].set_ylabel('Average Total Nights')
    axes[3, 0].set_xticks([0, 1])
    axes[3, 0].set_xticklabels(['Non-Repeated', 'Repeated'])

    # 8. Correlation Heatmap for Previous Reservations
    correlation_matrix = metrics["Correlation with Previous Reservations"].to_frame()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[3, 1])
    axes[3, 1].set_title('Correlation with Previous Reservations')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title space

    # Save figure if a path is provided
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()

    # Additional Plot: Market Segment Distribution for Repeated vs Non-Repeated Guests
    market_segment_repeated_counts = df.groupby(['market_segment_type', 'repeated_guest']).size().unstack().fillna(0)

    plt.figure(figsize=(12, 8))
    market_segment_repeated_counts.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], edgecolor='black')
    plt.title('Distribution of Repeated vs Non-Repeated Guests Across Market Segments')
    plt.xlabel('Market Segment')
    plt.ylabel('Number of Guests')
    plt.legend(['Non-Repeated', 'Repeated'], title='Guest Type')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Additional Plot: Distribution of Repeated Guests Across Market Segments
    repeated_guests_distribution = df[df['repeated_guest'] == 1]['market_segment_type'].value_counts(
        normalize=True) * 100

    plt.figure(figsize=(10, 6))
    repeated_guests_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Repeated Guests Among Market Segments')
    plt.xlabel('Market Segment')
    plt.ylabel('Percentage of Repeated Guests')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
