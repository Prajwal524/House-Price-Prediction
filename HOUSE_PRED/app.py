import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

houses = pd.read_csv('house_prices.csv')
X = houses.drop('SalePrice', axis=1)
y = houses['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def preprocess_input(medInc, houseAge, numRooms,
                    numBedrooms,population,
                    latitude,longitude):
    input_data = pd.DataFrame({
        'MedInc': [medInc],
        'HouseAge': [houseAge],
        'NumRooms': [numRooms],
        'NumBedrooms': [numBedrooms],
        'Population': [population],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    return input_data

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        medInc = float(data['medInc'])
        houseAge = float(data['houseAge'])
        numRooms = float(data['numRooms'])
        numBedrooms = float(data['numBedrooms'])
        population = float(data['population'])
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])

        input_data = preprocess_input(medInc,
                                       houseAge,
                                       numRooms,
                                       numBedrooms,
                                       population,
                                       latitude,
                                       longitude)

        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)