"""
Quick Start
https://facebook.github.io/prophet/docs/quick_start.html#python-api
"""

import pandas as pd
from prophet import Prophet

def main():
    print('main')
    # Python
    df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
    print(df.columns)
    if True:
        df = pd.read_csv('archive/GlobalLandTemperaturesByMajorCity.csv')
        df = df.rename(columns={"dt": "ds", "AverageTemperature": "y"})
        df = df[df['City'] == "New York"]
        df = df[df.y.notnull()]
        print(df.columns)
        print(df.head())

    # Python
    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365)
    print(future.tail())

    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    fig1 = m.plot(forecast)
    fig1.savefig('test.png')

    fig2 = m.plot_components(forecast)
    fig2.savefig('test2.png')


if __name__ == '__main__':
    main()
