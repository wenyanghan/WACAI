import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
def forward_selected(data,response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score,best_new_score = 0.0,0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,'+'.join(selected+[candidate]))
            score = smf.ols(formula,data).fit().rsquared_adj
            scores_with_candidates.append((score,candidate))
        scores_with_candidates.sort()
        print(scores_with_candidates)
        best_new_score,best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} +1".format(response,'+'.join(selected))
    model = smf.ols(formula,data).fit()
    return model
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
def feature_selection(X,y):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator,3,step = 1)
    selector = selector.fit(X, y)
    print(selector.support_)
    print(selector.ranking_)
    return selector
def get_data():
    path = "C:/Users/faye/Desktop/datatrain_new.csv"
    data = pd.read_csv(path)
    return data
#donnot need
def get_data_one():
    path = "C:/Users/faye/Desktop/datatrain_new.csv"
    data = pd.read_csv(path)
    del data['phone_gray_score']
    del data['contacts_loankeywords_redline_cnt']
    return data
def get_y_x(data):
    data_copy = data.copy()
    y = pd.DataFrame(data['labels'])
    del data_copy['labels']
    x = data_copy
    return x.values,y.values.ravel()
def test1():
    url = "http://data.princeton.edu/wws509/datasets/salary.dat"
    data = pd.read_csv(url,sep='\\s+')
    model = forward_selected(data, 'sl')
    print(model.model.formula)
    print(model.rsquared_adj)
    pred = (model.predict(data))
    print(pred)
    print(model.summary())  
def test2():
    data = get_data_one()
    model = forward_selected(data,'labels')
    print(model.model.formula)
    print(model.rsquared_adj)
    new_data = data.head(100)
    print(new_data)
    del new_data['labels']
    print(model.predict(new_data))
    data_2 = data.head(100)
#     del data_2['labels']
#     del data_2['honeypot_model_score']
    pred = model.predict(data_2)
    print(model.summary())
    print('sucess')
def test3():
    data = get_data()
    X,y = get_y_x(data)
    selector = feature_selection(X, y)
    print(selector.support_)
    print(selector.ranking_)
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
if __name__ == "__main__":
    test2()



