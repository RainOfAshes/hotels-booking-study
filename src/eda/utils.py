import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe, this is very specific to the dataset we have
    """
    df['has_children'] = df.no_of_children.apply(bool)

    df['total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']

    df['total_guests'] = df['no_of_adults'] + df['no_of_children']

    df['arrival_datetime'] = pd.to_datetime(df[['arrival_year', 'arrival_month', 'arrival_date']].rename(
        columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_date': 'day'}), errors='coerce')

    df['reservation_datetime'] = df['arrival_datetime'] - pd.to_timedelta(df['lead_time'], unit='D')

    df['no_previous_reservations'] = df.apply(
        lambda row: row['no_of_previous_cancellations'] + row['no_of_previous_bookings_not_canceled']
        if bool(row['repeated_guest']) else 0, axis=1
    )

    df = df.dropna(subset="arrival_datetime").sort_values(by="arrival_datetime").reset_index(drop=True)

    df['arrival_day_name'] = df['arrival_datetime'].dt.day_name()

    return df


if __name__ == "__main__":
    file_path = r"..\..\data\dataset.csv"
    df = pd.read_csv(file_path)
    df = add_engineered_features(df)
    print(df.columns)
