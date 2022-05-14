from fastapi import FastAPI
import uvicorn
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from pydantic import BaseModel
import gunicorn

# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    Distance_Covered : int
    Match_of_Route : int
    Fuel_Consumption : int
    Body_Characteristics : int
    Equipment_Sensors : int
    Efficiency : int
    

# Loading Dataset
truckData = pd.read_csv("truckData.csv")

df = pd.DataFrame(truckData)
# Header
names = ['Distance_Covered', 'Match_of_Route', 'Fuel_Consumption', 'Body_Characteristics', 'Equipment_Sensors', 'Efficiency', 'Rating']

X, Y = df.iloc[:, :-1], df.iloc[:, [-1]]

# Getting our Features and Targets
#X = truckData[:,0:6]
#Y = truckData[:,6]

# Creating and Fitting our Model
clf = DecisionTreeClassifier()
clf.fit(X,Y)

# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
	# Making the data in a form suitable for prediction
	test_data = [[
			data.Distance_Covered,
            data.Match_of_Route,
            data.Fuel_Consumption,
            data.Body_Characteristics,
            data.Equipment_Sensors,
            data.Efficiency
	]]
	
	# Predicting the Class
	result = clf.predict(test_data)[0]
	
	# Return the Result
	return result
