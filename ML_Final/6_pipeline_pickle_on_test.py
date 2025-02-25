import dill as pickle
import pandas as pd
from sklearn.metrics import roc_auc_score

def main():
    print('Bancruptcy Prediction on test')

    with open('model/bank_pipe.pkl', 'rb') as file:
        model = pickle.load(file)

    df_test = pd.read_csv('data/df_test.csv')
    X_test, y_test = df_test.drop('flag', axis=1), df_test['flag']

    pred = model['model'].predict_proba(X_test)
    score = roc_auc_score(y_test, pred[:, 1])
    print('roc_auc: ', score)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
