import dill as pickle
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector


def filter_data(df, columns_to_drop):
    import pandas as pd
    df_corr = df.copy()
    df_corr = df_corr.drop(columns_to_drop, axis=1)
    return df_corr


def delete_outliers(df, column):
    import pandas as pd
    df_corr = df.copy()

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return boundaries

    boundaries = calculate_outliers(df_corr[column])
    df_corr.loc[df[column] < boundaries[0], column] = round(boundaries[0])
    df_corr.loc[df[column] > boundaries[1], column] = round(boundaries[1])

    return df_corr


def device_brand_na_filling(df):
    import pandas as pd
    df_corr = df.copy()
    df_corr['device_brand'].fillna(df_corr['device_category'], inplace=True)

    return df_corr


def device_screen_engineering(df):
    import pandas as pd
    df_corr = df.copy()

    def screen_sqr(x):
        if not pd.isna(x):
            return int(x.split('x')[0]) * int(x.split('x')[1])
        else:
            return x

    df_corr['device_screen_sqr'] = df_corr['device_screen_resolution'].apply(screen_sqr)
    df_corr = df_corr.drop(['device_screen_resolution'], axis=1)
    return df_corr


def device_browser_engineering(df):
    import pandas as pd
    df_corr = df.copy()

    def shorten(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df_corr['device_browser_short'] = df_corr['device_browser'].apply(shorten)
    df_corr = df_corr.drop(['device_browser'], axis=1)

    return df_corr


def visit_hour_engineering(df):
    import pandas as pd
    df_corr = df.copy()

    def hour(x):
        if not pd.isna(x):
            return int(x.split(':')[0])
        else:
            return x

    df_corr['visit_hour'] = df_corr['visit_time'].apply(hour)
    df_corr['visit_day_night'] = df_corr['visit_hour'].apply(lambda x: 1 if (23 > x > 8) else 0)
    df_corr = df_corr.drop(['visit_time', 'visit_hour'], axis=1)

    return df_corr


def weekday_engineering(df):
    import pandas as pd
    df_corr = df.copy()

    df_corr['visit_date'] = pd.to_datetime(df_corr['visit_date'], format='%Y-%m-%d')
    df_corr['dayofweek'] = df_corr['visit_date'].dt.dayofweek
    df_corr = df_corr.drop(['visit_date'], axis=1)

    return df_corr


def utm_frequency(df, columns_list, freq_list):
    df_corr = df.copy()
    for col in range(len(columns_list)):
        df_corr[columns_list[col]].fillna('other', inplace=True)
        name = columns_list[col] + '_freq'
        df_corr[name] = df_corr[columns_list[col]].apply(lambda x: freq_list[col][x])

    return df_corr


def main():
    print('Target Action Prediction Pipeline')

    df = pd.read_csv('data/df_balanced.csv', index_col=0)

    with open('model/freq.pkl', 'rb') as file:
        freq_list = pickle.load(file)

    x = df.drop('target', axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    columns_to_drop = [
        'session_id',
        'client_id',
        'utm_keyword',
        'device_os'
    ]

    utm_freq_cols = ['utm_source', 'utm_campaign']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=0.005))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    model = GradientBoostingClassifier(random_state=42, max_features='sqrt', n_estimators=3000)

    pipe = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data, kw_args={"columns_to_drop": columns_to_drop})),
        ('device_brand_na_filling', FunctionTransformer(device_brand_na_filling)),
        ('device_screen_engineering', FunctionTransformer(device_screen_engineering)),
        ('outliers_filter', FunctionTransformer(delete_outliers, kw_args={"column": 'device_screen_sqr'})),
        ('device_browser_engineering', FunctionTransformer(device_browser_engineering)),
        ('visit_hour_engineering', FunctionTransformer(visit_hour_engineering)),
        ('weekday_engineering', FunctionTransformer(weekday_engineering)),
        ('utm_frequency_engineering', FunctionTransformer(utm_frequency,
                                                          kw_args={"columns_list": utm_freq_cols,
                                                                   "freq_list": freq_list})),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipe.fit(x_train, y_train)
    pred = pipe.predict_proba(x_test)
    score = roc_auc_score(y_test, pred[:, 1])
    print(f'model: {type(pipe.named_steps["classifier"]).__name__}, roc_auc: {score:.4f}')

    with open('model/cars_pipe.pkl', 'wb') as file:
        pickle.dump({
            'model': pipe,
            'metadata': {
                'name': 'target action prediction model',
                'author': 'Yana Eliseeva',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'roc_auc': score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
