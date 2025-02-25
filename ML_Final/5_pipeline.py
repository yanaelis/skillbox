import dill as pickle
import pandas as pd
import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


def filter_data(df, columns_to_drop):
    import pandas as pd
    df_corr = df.copy()
    df_corr = df_corr.drop(columns_to_drop, axis=1)
    return df_corr


def transform_to_int8(df, keep_columns=None):
    import pandas as pd
    df_corr = df.copy()
    if keep_columns is not None:
        df_sep = df_corr[keep_columns]
        df_corr = df_corr.drop(keep_columns, axis=1).astype('int8')
        df_corr[keep_columns] = df_sep
    else:
        df_corr = df_corr.astype('int8')
    return df_corr


def no_delays_rate(df, columns, name):
    import pandas as pd
    df_corr = df.copy()

    def count_zeros_rate(row, columns):
        count = 0
        for col in columns:
            count += row[col]
        return count / row['rn']

    df_corr[name] = df_corr.apply(lambda x: count_zeros_rate(x, columns), axis = 1)
    return df_corr

def no_delays_feat(df, columns, name):
    import pandas as pd
    df_corr = df.copy()

    def count_zeros_feat(row, columns):
        count = 0
        for col in columns:
            count += row[col]
        return count
    df_corr[name] = df_corr.apply(lambda x: count_zeros_feat(x, columns), axis = 1)
    return df_corr

def all_flags_rate(df, columns):
    import pandas as pd
    df_corr = df.copy()
    for col in columns:
        name = col + '_rate'
        df_corr[name] = df_corr.apply(lambda x: (x[col] / x['rn']), axis=1)

    return df_corr

def add_feature_from_pair(df, pairs, operation):
    import pandas as pd
    df_corr = df.copy()

    for pair in pairs:
        if operation == 'minus':
            name_minus = pair[0] + '_minus_' + pair[1]
            df_corr[name_minus] = df_corr.apply(lambda x: ((x[pair[0]]) - (x[pair[1]])).astype('int8'), axis = 1)
        elif operation == 'plus_per_rn':
            name_plus_rate = pair[0] + '_plus_' + pair[1] + '_rate'
            df_corr[name_plus_rate] = df_corr.apply(lambda x: (((x[pair[0]]) + (x[pair[1]]))/ x['rn']), axis = 1)
        elif operation == 'minus_per_rn':
            name_minus_rate = pair[0] + '_minus_' + pair[1] + '_rate'
            df_corr[name_minus_rate] = df_corr.apply(lambda x: (((x[pair[0]]) - (x[pair[1]])) / x['rn']), axis = 1)
        else:
            print('Wrong operation name')
    return df_corr


class EnsembleClassifier:
    def __init__(self, models):
        # Инициализация классификаторов
        self.models = models

    def fit(self, X, y):
        # Обучение всех классификаторов
        for model in self.models:
            print(f"Обучение {model}...")
            model.fit(X, y)

        self.fitted_ = True  #  осчастливим is_fitted, чтобы не было ворнингов
        return self

    def predict_proba(self, X):
        import numpy as np
        # Получение вероятностей для каждого классификатора
        probas = []
        for model in self.models:
            print(f"Предсказание {model}...")
            prob = model.predict_proba(X)
            probas.append(prob)

        # Усреднение вероятностей
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def predict(self, X):
        import numpy as np
        # Получение финальных предсказаний на основе усредненных вероятностей
        avg_proba = self.predict_proba(X)
        return np.argmax(avg_proba, axis=1)


def main():
    print('Bancruptcy Prediction Pipeline')

    df = pd.read_csv('data/df_done.csv')

    df_train, df_test = train_test_split(df, stratify=df['flag'], test_size=0.2, random_state=42)

    X_train, y_train = df_train.drop('flag', axis=1), df_train['flag']
    X_test, y_test = df_test.drop('flag', axis=1), df_test['flag']

    columns_to_drop = ['id']

    flags_no_delays = ['is_zero_loans5', 'is_zero_loans530',
                       'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90']

    flags_no_delays_over_530 = ['is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90']

    other_flags = ['is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit', 'pclose_flag', 'fclose_flag']
    flags = other_flags + flags_no_delays

    # генерация новых фичей - пары для преобразований
    features_for_subtraction = [['enc_paym_10_3', 'is_zero_loans530'],
                                ['is_zero_loans530', 'enc_paym_21_3']]

    features_for_sum_per_rn = [['pre_maxover2limit_infrequent_sklearn', 'enc_paym_3_3'],
                               ['enc_paym_17_infrequent_sklearn', 'enc_paym_4_3'],
                               ['enc_paym_13_infrequent_sklearn', 'enc_paym_4_3'],
                               ['enc_paym_4_3', 'enc_paym_1_1'],
                               ['enc_paym_4_3', 'enc_paym_12_1'],
                               ['enc_paym_4_3', 'pclose_flag_rate'],
                               ['enc_paym_1_infrequent_sklearn', 'enc_paym_6_3'],
                               ['fclose_flag_rate', 'enc_paym_4_3'],
                               ['pre_util_6', 'enc_loans_credit_type_5']
                               ]

    features_for_difference_per_rn = [['enc_paym_17_infrequent_sklearn', 'enc_paym_4_3'],
                                      ['enc_paym_9_3', 'pre_maxover2limit_17'],
                                      ['enc_paym_17_3', 'is_zero_loans5'],
                                      ['pre_util_5', 'enc_paym_5_3']
                                      ]

    # Второй проход - после повторной генерации фичей и отбора по максимальной корреляции
    # plus rate
    features_for_sum_per_rn_second_time = [['pre_util_4', 'enc_paym_10_3_minus_is_zero_loans530'],
                                           ['enc_paym_10_3_minus_is_zero_loans530', 'pre_util_5'],
                                           ['fclose_flag_rate', 'enc_paym_10_3_minus_is_zero_loans530'],
                                           ['enc_paym_0_1', 'enc_paym_1_infrequent_sklearn_plus_enc_paym_6_3_rate'],
                                           ['enc_paym_10_3_minus_is_zero_loans530',
                                            'pre_maxover2limit_infrequent_sklearn_plus_enc_paym_3_3_rate'],
                                           ['is_zero_loans530_minus_enc_paym_21_3', 'pre_loans_credit_limit_2'],
                                           ['enc_paym_4_3_plus_enc_paym_12_1_rate', 'enc_paym_0_1']
                                           ]

    # minus rate
    features_for_difference_per_rn_second_time = [['enc_paym_0_1', 'is_zero_loans530_minus_enc_paym_21_3'],
                                                  ['enc_paym_10_3_minus_is_zero_loans530', 'pre_util_5'],
                                                  ['enc_paym_10_3_minus_is_zero_loans530', 'pre_util_4'],
                                                  ['is_zero_loans530_minus_enc_paym_21_3', 'fclose_flag_rate'],
                                                  ['pre_util_5_minus_enc_paym_5_3_rate', 'enc_loans_credit_type_5']
                                                  ]

    # фичи, которые не нужно преобразовывать в int8
    rate_cols = ['no_delays_rate',
                 'no_delays',
                 'no_delays_over_530',
                 'is_zero_util_rate',
                 'is_zero_over2limit_rate',
                 'is_zero_maxover2limit_rate',
                 'pclose_flag_rate',
                 'fclose_flag_rate',
                 'is_zero_loans5_rate',
                 'is_zero_loans530_rate',
                 'is_zero_loans3060_rate',
                 'is_zero_loans6090_rate',
                 'is_zero_loans90_rate',
                 'enc_paym_10_3_minus_is_zero_loans530',
                 'is_zero_loans530_minus_enc_paym_21_3',
                 'pre_maxover2limit_infrequent_sklearn_plus_enc_paym_3_3_rate',
                 'enc_paym_17_infrequent_sklearn_plus_enc_paym_4_3_rate',
                 'enc_paym_13_infrequent_sklearn_plus_enc_paym_4_3_rate',
                 'enc_paym_4_3_plus_enc_paym_1_1_rate',
                 'enc_paym_4_3_plus_enc_paym_12_1_rate',
                 'enc_paym_4_3_plus_pclose_flag_rate_rate',
                 'enc_paym_1_infrequent_sklearn_plus_enc_paym_6_3_rate',
                 'fclose_flag_rate_plus_enc_paym_4_3_rate',
                 'pre_util_6_plus_enc_loans_credit_type_5_rate',
                 'enc_paym_17_infrequent_sklearn_minus_enc_paym_4_3_rate',
                 'enc_paym_9_3_minus_pre_maxover2limit_17_rate',
                 'enc_paym_17_3_minus_is_zero_loans5_rate',
                 'pre_util_5_minus_enc_paym_5_3_rate',
                 'pre_util_4_plus_enc_paym_10_3_minus_is_zero_loans530_rate',
                 'enc_paym_10_3_minus_is_zero_loans530_plus_pre_util_5_rate',
                 'fclose_flag_rate_plus_enc_paym_10_3_minus_is_zero_loans530_rate',
                 'enc_paym_0_1_plus_enc_paym_1_infrequent_sklearn_plus_enc_paym_6_3_rate_rate',
                 'enc_paym_10_3_minus_is_zero_loans530_plus_pre_maxover2limit_infrequent_sklearn_plus_enc_paym_3_3_rate_rate',
                 'is_zero_loans530_minus_enc_paym_21_3_plus_pre_loans_credit_limit_2_rate',
                 'enc_paym_4_3_plus_enc_paym_12_1_rate_plus_enc_paym_0_1_rate',
                 'enc_paym_0_1_minus_is_zero_loans530_minus_enc_paym_21_3_rate',
                 'enc_paym_10_3_minus_is_zero_loans530_minus_pre_util_5_rate',
                 'enc_paym_10_3_minus_is_zero_loans530_minus_pre_util_4_rate',
                 'is_zero_loans530_minus_enc_paym_21_3_minus_fclose_flag_rate_rate',
                 'pre_util_5_minus_enc_paym_5_3_rate_minus_enc_loans_credit_type_5_rate']

    models = [lgb.LGBMClassifier(learning_rate=0.03, n_estimators= 3000, max_depth= 15, verbosity = -1, random_state = 42),
              lgb.LGBMClassifier(learning_rate=0.03, n_estimators=3000, max_depth=12, verbosity=-1, random_state = 42),
              xgb.XGBClassifier(learning_rate=0.05, n_estimators=3000, max_depth=5, random_state = 42),
              xgb.XGBClassifier(learning_rate=0.01, n_estimators=3000, max_depth=9, random_state=42),
              CatBoostClassifier(verbose=0, learning_rate=0.07, iterations=2000, depth=9, random_state=42),
              CatBoostClassifier(verbose=0, learning_rate=0.05, iterations=2000, depth=9, random_state=42)
              ]


    pipe = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data, kw_args={"columns_to_drop": columns_to_drop})),
        ('transform_to_int8_1', FunctionTransformer(transform_to_int8)),
        ('no_delays_rate', FunctionTransformer(no_delays_rate, kw_args={"columns": flags_no_delays, "name":'no_delays_rate'})),
        ('no_delays_over_530_rate', FunctionTransformer(no_delays_rate, kw_args={"columns": flags_no_delays_over_530, "name": 'no_delays_over_530'})),
        ('no_delays_feat', FunctionTransformer(no_delays_feat, kw_args={"columns": flags_no_delays, "name": 'no_delays'})),
        ('all_flags_rate', FunctionTransformer(all_flags_rate, kw_args={"columns": flags})),
        ('add_feature_from_pair1', FunctionTransformer(add_feature_from_pair, kw_args={"pairs": features_for_subtraction, "operation": 'minus'})),
        ('add_feature_from_pair2', FunctionTransformer(add_feature_from_pair, kw_args={"pairs": features_for_sum_per_rn, "operation": 'plus_per_rn'})),
        ('add_feature_from_pair3', FunctionTransformer(add_feature_from_pair, kw_args={"pairs": features_for_difference_per_rn, "operation": 'minus_per_rn'})),
        ('add_feature_from_pair4', FunctionTransformer(add_feature_from_pair, kw_args={"pairs": features_for_sum_per_rn_second_time, "operation": 'plus_per_rn'})),
        ('add_feature_from_pair5', FunctionTransformer(add_feature_from_pair, kw_args={"pairs": features_for_difference_per_rn_second_time, "operation": 'minus_per_rn'})),
        ('transform_to_int8_2', FunctionTransformer(transform_to_int8, kw_args={"keep_columns": rate_cols})),
        ('classifier', EnsembleClassifier(models))
    ])


    pipe.fit(X_train, y_train)
    pred = pipe.predict_proba(X_test)
    score = roc_auc_score(y_test, pred[:, 1])
    print(f'model: {type(pipe.named_steps["classifier"]).__name__}, roc_auc: {score:.4f}')

    with open('model/bank_pipe.pkl', 'wb') as file:
        pickle.dump({
            'model': pipe,
            'metadata': {
                'name': 'target prediction model',
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
