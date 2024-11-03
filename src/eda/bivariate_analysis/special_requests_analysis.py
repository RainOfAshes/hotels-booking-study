import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def analyze_special_requests_influence(df):
    """
    Analyzes the influence of special requests on booking cancellations and calculates various
    metrics related to special requests in bookings.

    Args:
        df (pd.DataFrame): The booking data DataFrame.

    Returns:
        dict: A dictionary with calculated metrics for special requests and their impact on cancellations.
    """
    # Cancellation rate by number of special requests
    cancellation_rate_by_requests = df.groupby('no_of_special_requests')['booking_status'].apply(
        lambda x: (x == 'Canceled').mean() * 100)

    # Average previous reservations by number of special requests
    avg_previous_reservations_by_requests = df.groupby('no_of_special_requests')['no_previous_reservations'].mean()

    # Average total guests by number of special requests
    avg_total_guests_by_requests = df.groupby('no_of_special_requests')['total_guests'].mean()

    # Percentage of bookings with children by number of special requests
    has_children_by_requests = df.groupby('no_of_special_requests')['has_children'].mean() * 100

    # Average price per room by number of special requests
    avg_price_per_room_by_requests = df.groupby('no_of_special_requests')['avg_price_per_room'].mean()

    # Percentage of bookings requiring parking by number of special requests
    car_parking_by_requests = df.groupby('no_of_special_requests')['required_car_parking_space'].mean() * 100

    # Average total nights by number of special requests
    avg_total_nights_by_requests = df.groupby('no_of_special_requests')['total_nights'].mean()

    # Average lead time by number of special requests
    avg_lead_time_by_requests = df.groupby('no_of_special_requests')['lead_time'].mean()

    # Percentage of bookings with and without special requests
    percent_special_requests = df['no_of_special_requests'].apply(bool).value_counts(normalize=True) * 100

    # Cancellation percentage with special requests
    percent_cancellation_with_requests = df[df['no_of_special_requests'].apply(bool)]['booking_status'].value_counts(
        normalize=True) * 100

    # Cancellation percentage without special requests
    percent_cancellation_without_requests = df[~df['no_of_special_requests'].apply(bool)][
                                                'booking_status'].value_counts(normalize=True) * 100

    # Average special requests by market segment
    avg_special_requests_by_segment = df.groupby('market_segment_type')['no_of_special_requests'].mean()

    # Compile results into a dictionary
    return {
        "Cancellation Rate by Special Requests (%)": cancellation_rate_by_requests,
        "Avg Previous Reservations by Special Requests": avg_previous_reservations_by_requests,
        "Avg Total Guests by Special Requests": avg_total_guests_by_requests,
        "Percentage with Children by Special Requests (%)": has_children_by_requests,
        "Avg Price per Room by Special Requests": avg_price_per_room_by_requests,
        "Percentage Requiring Parking by Special Requests (%)": car_parking_by_requests,
        "Avg Total Nights by Special Requests": avg_total_nights_by_requests,
        "Avg Lead Time by Special Requests": avg_lead_time_by_requests,
        "Percentage of Bookings with/without Special Requests (%)": percent_special_requests,
        "Cancellation with Special Requests (%)": percent_cancellation_with_requests,
        "Cancellation without Special Requests (%)": percent_cancellation_without_requests,
        "Avg Special Requests by Market Segment": avg_special_requests_by_segment
    }


def plot_special_requests_analysis(df, save_fig_as=None):
    """
    Generates a set of visualizations analyzing the influence of special requests on booking cancellations,
    based on a set of precomputed metrics.

    Args:
        df (pd.DataFrame): pandas dataframe
        save_fig_as (str, optional): The file path to save the plot as a PNG image. Defaults to None.
    """
    metrics = analyze_special_requests_influence(df)
    fig, axes = plt.subplots(5, 2, figsize=(18, 28))
    fig.suptitle('Analysis of Special Requests and Related Metrics', fontsize=20)

    # 1. Cancellation Rate by Requests (Bar Chart)
    axes[0, 0].bar(metrics["Cancellation Rate by Special Requests (%)"].index,
                   metrics["Cancellation Rate by Special Requests (%)"].values, color='salmon')
    axes[0, 0].set_title('Cancellation Rate by Number of Special Requests')
    axes[0, 0].set_xlabel('Number of Special Requests')
    axes[0, 0].set_ylabel('Cancellation Rate (%)')

    # 2. Average Previous Reservations by Requests (Line Chart)
    axes[0, 1].plot(metrics["Avg Previous Reservations by Special Requests"].index,
                    metrics["Avg Previous Reservations by Special Requests"].values, marker='o', color='teal')
    axes[0, 1].set_title('Average Previous Reservations by Number of Special Requests')
    axes[0, 1].set_xlabel('Number of Special Requests')
    axes[0, 1].set_ylabel('Average Previous Reservations')

    # 3. Average Total Guests by Requests (Line Chart)
    axes[1, 0].plot(metrics["Avg Total Guests by Special Requests"].index,
                    metrics["Avg Total Guests by Special Requests"].values, marker='o', color='skyblue')
    axes[1, 0].set_title('Average Total Guests by Number of Special Requests')
    axes[1, 0].set_xlabel('Number of Special Requests')
    axes[1, 0].set_ylabel('Average Total Guests')

    # 4. Percentage with Children by Requests (Bar Chart)
    axes[1, 1].bar(metrics["Percentage with Children by Special Requests (%)"].index,
                   metrics["Percentage with Children by Special Requests (%)"].values, color='gold')
    axes[1, 1].set_title('Percentage of Bookings with Children by Number of Special Requests')
    axes[1, 1].set_xlabel('Number of Special Requests')
    axes[1, 1].set_ylabel('Percentage with Children')

    # 5. Average Price per Room by Requests (Line Chart)
    axes[2, 0].plot(metrics["Avg Price per Room by Special Requests"].index,
                    metrics["Avg Price per Room by Special Requests"].values, marker='o', color='purple')
    axes[2, 0].set_title('Average Price per Room by Number of Special Requests')
    axes[2, 0].set_xlabel('Number of Special Requests')
    axes[2, 0].set_ylabel('Average Price per Room')

    # 7. Car Parking by Requests (Bar Chart)
    axes[3, 0].bar(metrics["Percentage Requiring Parking by Special Requests (%)"].index,
                   metrics["Percentage Requiring Parking by Special Requests (%)"].values, color='green')
    axes[3, 0].set_title('Percentage Requiring Car Parking by Number of Special Requests')
    axes[3, 0].set_xlabel('Number of Special Requests')
    axes[3, 0].set_ylabel('Percentage Requiring Parking')

    # 8. Average Total Nights by Requests (Line Chart)
    axes[3, 1].plot(metrics["Avg Total Nights by Special Requests"].index,
                    metrics["Avg Total Nights by Special Requests"].values, marker='o', color='navy')
    axes[3, 1].set_title('Average Total Nights by Number of Special Requests')
    axes[3, 1].set_xlabel('Number of Special Requests')
    axes[3, 1].set_ylabel('Average Total Nights')

    # 9. Average Lead Time by Requests (Line Chart)
    axes[4, 0].plot(metrics["Avg Lead Time by Special Requests"].index,
                    metrics["Avg Lead Time by Special Requests"].values, marker='o', color='indigo')
    axes[4, 0].set_title('Average Lead Time by Number of Special Requests')
    axes[4, 0].set_xlabel('Number of Special Requests')
    axes[4, 0].set_ylabel('Average Lead Time (Days)')

    # 10. Special Requests by Market Segment (Bar Chart)
    axes[4, 1].bar(metrics["Avg Special Requests by Market Segment"].index,
                   metrics["Avg Special Requests by Market Segment"].values, color='coral')
    axes[4, 1].set_title('Average Special Requests by Market Segment')
    axes[4, 1].set_xlabel('Market Segment')
    axes[4, 1].set_ylabel('Average Number of Special Requests')
    axes[4, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Optional save to file
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()

    # Additional Pie and Bar Charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Special Requests and Cancellation Analysis', fontsize=16)

    # Plot 1: Percent of Data with and without Special Requests (Pie Chart)
    axes[0].pie(metrics["Percentage of Bookings with/without Special Requests (%)"].values,
                labels=['With Special Requests', 'Without Special Requests'], autopct='%1.1f%%', startangle=90,
                colors=['blue', 'orange'])
    axes[0].set_title('Percentage of Bookings with Special Requests')

    # Plot 2: Percent of Cancellations with Special Requests
    axes[1].bar(metrics["Cancellation with Special Requests (%)"].index,
                metrics["Cancellation with Special Requests (%)"].values, color=['green', 'red'])
    axes[1].set_title('Booking Status with Special Requests')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].xaxis.set_major_locator(ticker.FixedLocator([0, 1]))
    axes[1].set_xticklabels(['Not Canceled', 'Canceled'])

    # Plot 3: Percent of Cancellations without Special Requests
    axes[2].bar(metrics["Cancellation without Special Requests (%)"].index,
                metrics["Cancellation without Special Requests (%)"].values, color=['green', 'red'])
    axes[2].set_title('Booking Status without Special Requests')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].xaxis.set_major_locator(ticker.FixedLocator([0, 1]))
    axes[2].set_xticklabels(['Not Canceled', 'Canceled'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save additional plots if needed
    if save_fig_as:
        plt.savefig(save_fig_as.replace('.png', '_additional.png'), format='png')

    plt.show()
