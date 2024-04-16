#Splitting the data into to 3 different sets
# 1- Training set containing 70% of data
# 2- Testing set containing 15% of data
# 3- Validation set containing 15% of data

# Importin necessary libraries especially rain_test_split from sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

# loading the dataset we need to perform operation on
dst = 'ce889_dataCollection.csv'
data = dst
data = pd.read_csv(dst)
 # appplying normalization on the whole data set
def nrmlzd(dta):
  normalized = (dta - dta.min()) / (dta.max() - dta.min())
  return  normalized
newdta = nrmlzd(data)

# defining our inputs and outputs columns in datset
# x as an input columns and y as output columns
x = newdta.iloc[ : ,:2].values
y = newdta.iloc[ : ,2:].values
# apllying train_test_stplit twice once on whole data n x and y we defined and we divide data into 70% training and 30% rest
x_train, x_test1, y_train, y_test1 = train_test_split(x,y, test_size = 0.3, shuffle = True)
# defining our data inform of list as this program works with that
trainingData = [x_train, y_train]

# deviding rest of data into 15% of testing and validation each and definig them so they can be loaded
x_test, x_vald, y_test, y_vald = train_test_split(x_test1, y_test1, test_size= 0.15, shuffle= True)
testingData = [x_test, y_test]
validationData = [x_vald, y_vald]
