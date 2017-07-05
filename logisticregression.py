from sklearn.linear_model import LogisticRegression
import pandas as pd
def get_data():
    path = "C:/Users/faye/Desktop/datatrain_new.csv"
    data = pd.read_csv(path)
    return data
def get_y_x(data):
    data_copy = data.copy()
    y = pd.DataFrame(data['labels'])
    del data_copy['labels']
    x = data_copy
    return x,y
    
def logistic():
    data = get_data()
    x,y = get_y_x(data)
    classifier = LogisticRegression()
    classifier.fit(x.values,y.values.ravel())
    return classifier

if __name__ =="__main__":
    model = logistic()
    data = get_data()
    test = data.head(100)
    del test['labels']
    p = model.predict_proba(test.values)
    print(p)