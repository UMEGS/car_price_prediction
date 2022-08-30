import pickle

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    data = pd.read_csv('static/data/clean_data.csv')
    company = sorted(data['company'].unique())
    year = sorted(data['year'].unique(), reverse=True)
    fuel_type = sorted(data['fuel_type'].unique())
    car_model = sorted(data['name'].unique())
    prediction = None

    if request.method == 'POST':
        company = request.form.get('company')
        year = request.form.get('year')
        fuel_type = request.form.get('fuelType')
        car_model = request.form.get('carModel')
        travelled = request.form.get('travelled')
        prediction = predict(company, year, fuel_type, car_model, travelled)
        print(company, year, fuel_type, car_model, travelled, prediction)


    return render_template('index.html',
                           company=company,
                           year=year,
                           fuel_type=fuel_type,
                           car_model=car_model,
                           prediction = prediction
                           )


def predict(company, year, fuel_type, car_model, travelled):
    data = pd.DataFrame(
        {'name': car_model, 'company': company, 'year': year, 'fuel_type': fuel_type, 'kms_driven': travelled},index=[0])
    pickle_in = open('static/model/LinearRegressionModel.pkl', 'rb')
    model = pickle.load(pickle_in)
    prediction = model.predict(data)
    return round(prediction[0])


if __name__ == '__main__':
    app.run()
