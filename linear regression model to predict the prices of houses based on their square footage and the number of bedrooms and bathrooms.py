import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cryptography
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
#linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms
#df means datafile 
def trainModel():
    training_df =pd.read_csv(r"C:\Users\Benwari Ezekiel\Documents\code\house-prices-advanced-regression-techniques\train.csv")
    test_df = pd.read_csv(r"C:\Users\Benwari Ezekiel\Documents\code\house-prices-advanced-regression-techniques\test.csv")
    print(training_df.head)
    print(training_df.describe())
    print(training_df.info())
    
    training_df['totalBathroom'] = (training_df['FullBath']+ training_df['BsmtFullBath']+(0.5 * training_df['BsmtHalfBath'])+(0.5 * training_df['HalfBath']))
    test_df['totalBathroom'] = (test_df['FullBath']+ test_df['BsmtFullBath']+(0.5 * test_df['BsmtHalfBath'])+(0.5 * test_df['HalfBath']))
    training_df['totalBathroom'] = training_df['totalBathroom'].fillna(0)
    test_df['totalBathroom'] = test_df['totalBathroom'].fillna(0)
    
    newTrainDf = training_df.dropna(how='all',axis=0)
    newTestDf = test_df.dropna(how='all',axis=0)
    x_train = newTrainDf[['LotArea','BedroomAbvGr','totalBathroom']]
    y_train = newTrainDf['SalePrice']

    x_test = newTestDf[['LotArea','BedroomAbvGr','totalBathroom']]
   #y_test = newTestDf['SalePrice']
    #x_test, x_train, y_test, y_train = train_test_split(train_size=0.2, random_state=42)
    #print(training_df)
    #print(test_df)
    #print(newTrainDf.head())
    #print(newTestDf)
    #print(x_train)
    #print(y_train)
    #print(x_train.shape)
    #print(y_train.shape)
    #print(y_train.isnull().sum())
    #if not x_train.empty and y_train.empty: 
    #missing_index = y_train[y_train.isnull()].index
    #if not missing_index.empty:
    x_train = x_train.drop(index = 1459)
    y_train = y_train.drop(index=1459)
       # y_train.drop(index = missing_index)
    #x_train = x_train.transpose()
    #y_train = y_train.transpose()
    try:
        model = LinearRegression()
        model.fit(x_train,y_train)
        
        y_pred = model.predict(x_test)
        #print(y_pred)
        mse =mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train,y_pred)
        print(f"mean square scrore{mse}")
        print(f"Rsquared score{r2}")
    except Exception as e:
         print(f'error with training {e}')
            
    plt.scatter(y_train, y_pred)  # Create a scatter plot of actual vs. predicted prices.
    plt.xlabel('Actual Prices')  # Label the x-axis.
    plt.ylabel('Predicted Prices')  # Label the y-axis.
    plt.title('Actual vs Predicted Prices')  # Set the title of the plot.
    plt.show()  # Display the plot.
    return predict(model)

def predict(model):
    LotArea = int(input("enter lotsze n square mter"))
    Bedroom = int(input('enter total number of bedrooms'))
    totalBathroom= int(input("enter total number of bathrooms"))
    x_test = np.array([[LotArea,Bedroom,totalBathroom]])
    prediction = model.predict(x_test)
    print(prediction)
    
trainModel()

