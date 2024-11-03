import matplotlib.pyplot as plt
import pandas as pd


def analyze_revenue_optimization(df: pd.DataFrame):
    """
    Analyzes revenue optimization strategies for hotels by calculating revenue metrics across
    various factors, including market segment, special requests, and repeated guests.

    Args:
        df (pd.DataFrame): The booking data DataFrame.

    Returns:
        dict: A dictionary containing calculated revenue metrics and differences across segments.
    """
    # Revenue by Market Segment (Only for Non-Canceled Bookings)
    revenue_by_segment = df[df['booking_status'] == 'Not_Canceled'].groupby('market_segment_type')['avg_price_per_room'].sum()

    # Revenue by Presence of Children (Only for Non-Canceled Bookings)
    revenue_by_children = df[df['booking_status'] == 'Not_Canceled'].groupby('has_children')['avg_price_per_room'].sum()

    # Revenue by Number of Special Requests (Only for Non-Canceled Bookings)
    revenue_by_requests = df[df['booking_status'] == 'Not_Canceled'].groupby('no_of_special_requests')['avg_price_per_room'].sum()

    # Total Revenue (Only for Non-Canceled Bookings)
    total_revenue = df[df['booking_status'] == 'Not_Canceled']['avg_price_per_room'].sum()

    # Monthly Revenue Difference for 2018 (Non-Canceled - Canceled)
    monthly_revenue_diff_2018 = df[df['arrival_year'] == 2018].groupby('arrival_month').apply(
        lambda x: x.loc[x['booking_status'] == 'Not_Canceled', 'avg_price_per_room'].sum() -
                  x.loc[x['booking_status'] == 'Canceled', 'avg_price_per_room'].sum(), include_groups=False
    )

    # Revenue Difference by Market Segment (Non-Canceled - Canceled)
    revenue_diff_by_segment = df.groupby('market_segment_type').apply(
        lambda x: x.loc[x['booking_status'] == 'Not_Canceled', 'avg_price_per_room'].sum() -
                  x.loc[x['booking_status'] == 'Canceled', 'avg_price_per_room'].sum(), include_groups=False
    )

    # Revenue Difference by Number of Special Requests (Non-Canceled - Canceled)
    revenue_diff_by_requests = df.groupby('no_of_special_requests').apply(
        lambda x: x.loc[x['booking_status'] == 'Not_Canceled', 'avg_price_per_room'].sum() -
                  x.loc[x['booking_status'] == 'Canceled', 'avg_price_per_room'].sum(), include_groups=False
    )

    # Revenue Difference by Repeated Guest Status (Non-Canceled - Canceled)
    revenue_diff_by_repeated_guests = df.groupby('repeated_guest').apply(
        lambda x: x.loc[x['booking_status'] == 'Not_Canceled', 'avg_price_per_room'].sum() -
                  x.loc[x['booking_status'] == 'Canceled', 'avg_price_per_room'].sum(), include_groups=False
    )

    # Compile results into a dictionary
    return {
        "Revenue by Segment": revenue_by_segment,
        "Revenue by Children Presence": revenue_by_children,
        "Revenue by Special Requests": revenue_by_requests,
        "Total Revenue": total_revenue,
        "Monthly Revenue Difference (2018)": monthly_revenue_diff_2018,
        "Revenue Difference by Segment": revenue_diff_by_segment,
        "Revenue Difference by Requests": revenue_diff_by_requests,
        "Revenue Difference by Repeated Guests": revenue_diff_by_repeated_guests
    }


def plot_revenue_optimization_metrics(df, save_fig_as=None):
    """
    Generates a set of plots to analyze revenue optimization metrics across various factors,
    including monthly revenue differences, market segments, special requests, and repeated guests.

    Args:
        df (pd.DataFrame): The booking data DataFrame.
        save_fig_as (str, optional): The file path to save the plot as a PNG image. Defaults to None.
    """
    # Calculate revenue metrics using the analysis function
    metrics = analyze_revenue_optimization(df)

    # Set up figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Plot 1: Monthly Revenue Difference for 2018 (Bar Plot)
    metrics['Monthly Revenue Difference (2018)'].plot(kind='bar', color="#2ecc71", ax=axes[0, 0])
    axes[0, 0].set_title("Monthly Revenue Difference (2018)")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Revenue Difference ($)")
    axes[0, 0].set_xticks(range(12))
    axes[0, 0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                               rotation=45)

    # Plot 2: Revenue Difference by Market Segment (Bar Plot)
    metrics['Revenue Difference by Segment'].plot(kind='bar', color="#3498db", ax=axes[0, 1])
    axes[0, 1].set_title("Revenue Difference by Market Segment")
    axes[0, 1].set_xlabel("Market Segment")
    axes[0, 1].set_ylabel("Revenue Difference ($)")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

    # Plot 3: Revenue Difference by Special Requests (Bar Plot)
    metrics['Revenue Difference by Requests'].plot(kind='bar', color="#9b59b6", ax=axes[1, 0])
    axes[1, 0].set_title("Revenue Difference by Special Requests")
    axes[1, 0].set_xlabel("Number of Special Requests")
    axes[1, 0].set_ylabel("Revenue Difference ($)")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

    # Plot 4: Revenue Difference for Repeated vs. Non-Repeated Guests (Bar Plot)
    revenue_diff_by_repeated_guests = metrics['Revenue Difference by Repeated Guests']
    revenue_diff_by_repeated_guests.index = ["Non-Repeated Guests", "Repeated Guests"]
    revenue_diff_by_repeated_guests.plot(kind='bar', color="#e74c3c", ax=axes[1, 1])
    axes[1, 1].set_title("Revenue Difference for Repeated vs Non-Repeated Guests")
    axes[1, 1].set_xlabel("Guest Type")
    axes[1, 1].set_ylabel("Revenue Difference ($)")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()

    # Save figure if save path is provided
    if save_fig_as:
        plt.savefig(save_fig_as, format='png')

    plt.show()

    # Additional Plot 1: Market Segment Revenue and Loss Distribution (Pie Chart)
    fig, ax = plt.subplots(figsize=(8, 8))
    total_values = [metrics["Total Revenue"], metrics["Revenue Difference by Segment"].sum()]
    ax.pie(total_values, labels=["Total Revenue (Non-Canceled)", "Total Loss (Canceled)"], autopct='%1.1f%%',
           startangle=140, colors=["#2ecc71", "#e74c3c"])
    ax.set_title("Total Revenue vs. Total Loss")
    plt.show()

    # Additional Plot 2: Revenue vs. Loss by Presence of Children (Bar Plot)
    children_df = pd.DataFrame({
        'Revenue': metrics["Revenue by Children Presence"],
        'Loss': [metrics["Revenue by Children Presence"][False] - metrics["Revenue by Children Presence"][True],
                 metrics["Revenue by Children Presence"][True]]
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    children_df.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=ax)
    ax.set_title("Revenue vs. Loss by Presence of Children")
    ax.set_xlabel("Has Children")
    ax.set_ylabel("Total Amount ($)")
    ax.legend(title="Type")
    plt.show()

    # Additional Plot 3: Monthly Revenue and Loss for 2018 (Line Plot)
    monthly_revenue_loss_df = pd.DataFrame({
        'Revenue': metrics['Monthly Revenue Difference (2018)'],
        'Loss': metrics['Monthly Revenue Difference (2018)'] * -1  # assuming the diff represents net gain/loss
    })
