from fastapi import FastAPI
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from pydantic import BaseModel

# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute


class request_body(BaseModel):
    Distance_Covered: int
    Match_of_Route: int
    Fuel_Consumption: int
    Body_Characteristics: int
    Equipment_Sensors: int
    Efficiency: int
    Distance_Covered2: int
    Match_of_Route2: int
    Fuel_Consumption2: int
    Body_Characteristics2: int
    Equipment_Sensors2: int
    Efficiency2: int
    Distance_Covered3: int
    Match_of_Route3: int
    Fuel_Consumption3: int
    Body_Characteristics3: int
    Equipment_Sensors3: int
    Efficiency3: int


# Loading Dataset
truckData = pd.read_csv("truckData.csv")

df = pd.DataFrame(truckData)
# Header
names = ['Distance_Covered', 'Match_of_Route', 'Fuel_Consumption',
    'Body_Characteristics', 'Equipment_Sensors', 'Efficiency', 'Rating']

X, Y = df.iloc[:, :-1], df.iloc[:, [-1]]


# Creating and Fitting our Model
clf = DecisionTreeClassifier()
clf.fit(X, Y)

# Creating an Endpoint to receive the data
# to make prediction on.


@app.post('/predict')
def predict(data: request_body):
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
