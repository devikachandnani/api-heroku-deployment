"""

Step 1 - building the linear regression model which uses Biking and Smoking as features to predict percentage
of the population that may have heart disease in an imaginary town of 500 samples.

"""

#import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model

#load dataset
df = pd.read_csv('heart_data.csv')
print(df.head())

df = df.drop("Unnamed: 0", axis=1)

sns.lmplot(x='biking', y='heart.disease', data=df)  
sns.lmplot(x='smoking', y='heart.disease', data=df)  

x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

#split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

#create linear regression object
model = linear_model.LinearRegression()
model.fit(X_train, y_train) 
print(model.score(X_train, y_train)) 

prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)

#store as pickle file
import pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[20.1, 56.3]]))

