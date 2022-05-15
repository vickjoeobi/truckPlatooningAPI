from sre_parse import State
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from pydantic import BaseModel

# Creating FastAPI instance
app = FastAPI()

# CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# creating class to define the request body
# and the type hints of each attribute


class request_body(BaseModel):
    TruckName: str
    Distance_Covered: str
    Match_of_Route: str
    Fuel_Consumption: str
    Body_Characteristics: str
    Equipment_Sensors: str
    Efficiency: str
    TruckName2: str
    Distance_Covered2: str
    Match_of_Route2: str
    Fuel_Consumption2: str
    Body_Characteristics2: str
    Equipment_Sensors2: str
    Efficiency2: str
    TruckName3: str
    Distance_Covered3: str
    Match_of_Route3: str
    Fuel_Consumption3: str
    Body_Characteristics3: str
    Equipment_Sensors3: str
    Efficiency3: str


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
    
    # converting the data to int individually
    Disatnce_Covered = int(data.Distance_Covered)
    Match_of_Route = int(data.Match_of_Route)
    Fuel_Consumption = int(data.Fuel_Consumption)
    Body_Characteristics = int(data.Body_Characteristics)
    Equipment_Sensors = int(data.Equipment_Sensors)
    Efficiency = int(data.Efficiency)
    Disatnce_Covered2 = int(data.Distance_Covered2)
    Match_of_Route2 = int(data.Match_of_Route2)
    Fuel_Consumption2 = int(data.Fuel_Consumption2)
    Body_Characteristics2 = int(data.Body_Characteristics2)
    Equipment_Sensors2 = int(data.Equipment_Sensors2)
    Efficiency2 = int(data.Efficiency2)
    Disatnce_Covered3 = int(data.Distance_Covered3)
    Match_of_Route3 = int(data.Match_of_Route3)
    Fuel_Consumption3 = int(data.Fuel_Consumption3)
    Body_Characteristics3 = int(data.Body_Characteristics3)
    Equipment_Sensors3 = int(data.Equipment_Sensors3)
    Efficiency3 = int(data.Efficiency3)
    
    # Making the data in a form suitable for prediction
    test_data = [[
        Disatnce_Covered,
        Match_of_Route,
        Fuel_Consumption,
        Body_Characteristics,
        Equipment_Sensors,
        Efficiency
    ]]
    test_data2 = [[
        Disatnce_Covered2,
        Match_of_Route2,
        Fuel_Consumption2,
        Body_Characteristics2,
        Equipment_Sensors2,
        Efficiency2
    ]]
    test_data3 = [[
        Disatnce_Covered3,
        Match_of_Route3,
        Fuel_Consumption3,
        Body_Characteristics3,
        Equipment_Sensors3,
        Efficiency3
    ]]

    # Making the prediction
    result1 = clf.predict(test_data)[0]
    result2 = clf.predict(test_data2)[0]
    result3 = clf.predict(test_data3)[0]

    # Map the prediction to the rating
    rating1 = 0
    rating2 = 0
    rating3 = 0

    if result1 == 'Highest':
        rating1 = 4
    elif result1 == 'High':
        rating1 = 3
    elif result1 == 'low':
        rating1 = 2
    elif result1 == 'lowest':
        rating1 = 1
    else:
        rating1 = 0

    if result2 == 'Highest':
        rating2 = 4
    elif result2 == 'High':
        rating2 = 3
    elif result2 == 'low':
        rating2 = 2
    elif result2 == 'lowest':
        rating2 = 1
    else:
        rating2 = 0

    if result3 == 'Highest':
        rating3 = 4
    elif result3 == 'High':
        rating3 = 3
    elif result3 == 'low':
        rating3 = 2
    elif result3 == 'lowest':
        rating3 = 1
    else:
        rating3 = 0

    # Return the max rating
    if (rating1 >= rating2) and (rating1 >= rating3):
        return data.TruckName + ':' + ' ' + result1
    elif (rating2 >= rating1) and (rating2 > rating3):
        return data.TruckName2 + ':' + ' ' + result2
    elif (rating3 >= rating1) and (rating3 >= rating2):
        return data.TruckName3 + ':' + ' ' + result3
    else:
        return data.TruckName + ':' + ' ' + result1
