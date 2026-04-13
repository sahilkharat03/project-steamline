import pandas as pd

def load_and_clean_data(path):

    df = pd.read_csv(path)

    # ✅ Clean column names
    df.columns = df.columns.str.strip()

    # ✅ Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # ✅ Convert numeric columns safely
    cols = [
        'Children in HHS Care',
        'Children transferred out of CBP custody',
        'Children discharged from HHS Care'
    ]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ Sort & index
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # ✅ Fill missing BEFORE features
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df