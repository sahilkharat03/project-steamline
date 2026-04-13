def create_features(df):

    # ✅ only small lag (keep data)
    df['lag_1'] = df['Children in HHS Care'].shift(1)

    # ❌ REMOVE lag_7 and rolling (they kill data)

    df['net_flow'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']

    df['day_of_week'] = df.index.dayofweek

    # ✅ fill safely
    df = df.fillna(method='bfill')

    return df