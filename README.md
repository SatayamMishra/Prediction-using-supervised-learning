# Name: Satyam Mishra

# Task 1: Predicting using Supervised Machine Learning

# GRIP@SparkFoundation

In this Task 1 I predict the percentage of an student based om the no of study hours.

This is the Simple Linear Regression Task as it involves just two variables.

# Importing all required libraries.  
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Step 1- Reading the data from source

url="http://bit.ly/w-data"
student_data=pd.read_csv(url) 
print("Data Import Succesfully")
student_data.head(10)

# Step 2- Plots Our data points on 2d Graph

# Plotting the distribution of score
student_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scores')
plt.show()

# Step 3- Now Devided the data into "Atributes" and "Labels"

x=student_data.iloc[:,:1].values
y=student_data.iloc[:,1:2].values

# Step 4- Split the data into the Training and Testing sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Step 5- Now Train our Algorithm

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
print("Training Complete")

# Step 6- Plotting the Regression line for the test data

# Plotting the regression line
line = lm.coef_*x+lm.intercept_

# Plotting for the test data
plt.scatter(x, y,color="gray")
plt.plot(x, line,color="Green");
plt.show()

# Step 6- Pridicting the Model

y_pred=lm.predict(x_test)

# Step 7- Comparing Actual Vs Prediction

df =pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})  
df 

# Step 8- Test the model

hours=9.25
test=np.array([hours])
test=test.reshape(-1,1)
own_pred=lm.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

# Step 9- Evaluate the Performance of the model

from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
