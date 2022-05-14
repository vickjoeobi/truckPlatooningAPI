from sre_parse import State
from fastapi import FastAPI
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from pydantic import BaseModel

# Creating FastAPI instance
app = FastAPI()

#creating class to define the request body
# and the type hints of each attribute


class request_body(BaseModel):
    TruckName: str
    Distance_Covered: int
    Match_of_Route: int
    Fuel_Consumption: int
    Body_Characteristics: int
    Equipment_Sensors: int
    Efficiency: int
    TruckName2: str
    Distance_Covered2: int
    Match_of_Route2: int
    Fuel_Consumption2: int
    Body_Characteristics2: int
    Equipment_Sensors2: int
    Efficiency2: int
    TruckName3: str
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
names = ['Distance_Covered', 'Match_of_Route', 'Fuel_Consumption', 'Body_Characteristics', 'Equipment_Sensors', 'Efficiency', 'Rating']

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
        data.Efficiency,
    ],
    [
        data.Distance_Covered2,
        data.Match_of_Route2,
        data.Fuel_Consumption2,
        data.Body_Characteristics2,
        data.Equipment_Sensors2,
        data.Efficiency2,
    ],
    [
        data.Distance_Covered3,
        data.Match_of_Route3,
        data.Fuel_Consumption3,
        data.Body_Characteristics3,
        data.Equipment_Sensors3,
        data.Efficiency3,
    ]]

    # Making the prediction
    result1 = clf.predict(test_data)[0]
    result2 = clf.predict(test_data)[1]
    result3 = clf.predict(test_data)[2]
    
    # Map the prediction to the rating
    
    rating1 = 4 if result1[0] == 'Highest' else 3 if result1[0] == 'High' else 2 if result1[0] == 'low' else 1 if result1[0] == 'lowest' else 0
    rating2 = 4 if result1[0] == 'Highest' else 3 if result1[
        0] == 'High' else 2 if result1[0] == 'low' else 1 if result1[0] == 'lowest' else 0
    rating3 = 4 if result1[0] == 'Highest' else 3 if result1[
        0] == 'High' else 2 if result1[0] == 'low' else 1 if result1[0] == 'lowest' else 0

    # Return the max rating
    if rating1 > rating2 and rating1 > rating3:
        return result1
    elif rating2 > rating1 and rating2 > rating3:
        return result2
    elif rating3 > rating1 and rating3 > rating2:
        return result3
    else:
        return result1