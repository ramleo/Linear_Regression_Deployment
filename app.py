import pickle
import werkzeug
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from module.user_defined_functions import prep_data

# Creating app object

app = Flask(__name__)

# Providing min and max range as this is linear regression model
# this model cannot extrapolate hence values provided must be
# within min and max range

values_range = {
 'season':{'min': 1, 'max': 4},
 'yr': {'min': 0, 'max': 1},
 'mnth': {'min': 1, 'max': 12},
 'holiday': {'min': 0, 'max': 1},
 'weekday': {'min': 0, 'max': 6},
 'workingday': {'min': 0, 'max': 1},
 'weathersit': {'min': 1, 'max': 3},
 'temp': {'min': 2.4243464, 'max': 34.815847},
 'atemp': {'min': 3.9534800000000003, 'max': 42.0448},
 'hum': {'min': 0.0, 'max': 97.25},
 'windspeed': {'min': 1.5002438999999999, 'max': 34.000021000000004}
}

## Predicting Test data## Checking the model with single data point from test dataset# Loading required models

ohe = pickle.load(open('V2_OneHotEncoder.pkl', 'rb'))
minmax = pickle.load(open('V2_MinmaxScaler.pkl', 'rb'))
lr_model = pickle.load(open('V2_LinearModel.pkl', 'rb'))

# Cloumns selected by sklearn RFE function

sel_cols = ['yr', 'spring', 'Clear', 'LightRain_LightSnow', 'Mist_Cloudy', 'Apr', 'Dec', 'Feb', 'Jan', 'Mar', 'Nov', 'Mon']

# Features dummy encoded by sklearn One Hot Encoder

enc_feat = ['fall', 'spring', 'summer', 'winter', 'Clear', 'LightRain_LightSnow', 'Mist_Cloudy', 'Apr', 'Aug', 'Dec',
            'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep', 'Fri', 'Mon', 'Sat', 'Sun', 'Thur', 'Tue', 'Wed']


# Exposing the below code to localhost:5000

@app.route('/api', methods=['POST'])
def pred_testquerry():
    df = pd.DataFrame()
    content = request.json
    errors = []
    for i in content:
        if i in values_range:
            min_range = values_range[i]['min']
            max_range = values_range[i]['max']
            value = content[i]
            if value < min_range or value > max_range:
                errors.append(f'The values should be between {min_range} and {max_range}.')
        else:
            errors.append(f'Unexpected field: {i}.')

    for i in values_range:
        if i not in content:
            errors.append(f'Missing value: {i}')

    if len(errors) < 1:
        df['season'] = pd.Series(content['season'])
        df['yr'] = pd.Series(content['yr'])
        df['mnth'] = pd.Series(content['mnth'])
        df['holiday'] = pd.Series(content['holiday'])
        df['weekday'] = pd.Series(content['weekday'])
        df['workingday'] = pd.Series(content['workingday'])
        df['weathersit'] = pd.Series(content['weathersit'])
        df['temp'] = pd.Series(content['temp'])
        df['atemp'] = pd.Series(content['atemp'])
        df['hum'] = pd.Series(content['hum'])
        df['windspeed'] = pd.Series(content['windspeed'])

        def y_pred(X_data=df, minmax=minmax, model=lr_model, onehotencoder=ohe):

            # Preprocessing test data
            X_data = prep_data(X_data=X_data)

            # Transforming into onehot encoders
            tmp = onehotencoder.transform(X_data[['season', 'weathersit', 'mnth', 'weekday']])

            # Creating data frame of top 15 features selected from RFE
            X_data.reset_index(inplace=True)
            tmp1 = pd.DataFrame(tmp, columns=enc_feat)
            X_data = pd.concat([X_data, tmp1], axis=1)
            X_data.drop(['season', 'weathersit', 'mnth', 'weekday'], axis=1, inplace=True)
            X_data = X_data[sel_cols]

            # Transform test data
            X_data = minmax.transform(X_data)

            # Fitting Linear Regression model
            pred = model.predict(X_data)

            return pred

        prediction = y_pred()
        bike_count = float(prediction)
        response = {'bike_count': bike_count, 'errors': errors}

    else:
        response = {'errors': errors}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)