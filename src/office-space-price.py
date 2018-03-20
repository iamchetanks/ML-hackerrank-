import pandas as pd
from sklearn import linear_model

def read_data():
    df = pd.read_csv("../data/office-space-price.txt", header=None, names=['f1','f2','price'], sep=' ')
    #print(df['f1'])
    return df

def regression(df):
    X_train = []
    reg = linear_model.LinearRegression()
    for i in zip(df['f1'][0:-4],df['f2'][0:-4]):
        X_train.append(list(i))
    y_train = df['price'][0:-4]
    reg.fit(X_train, y_train)
    X_test = []
    for i in zip(df['f1'][-4:], df['f2'][-4:]):
        X_test.append(list(i))

    print("coeff",reg.coef_)
    print(reg.predict(X_train))




def main():
    df = read_data()
    regression(df)

if __name__ == "__main__":
    main()