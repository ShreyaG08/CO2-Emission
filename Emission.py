# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:45:58 2025

@author: Dell
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Dell/Desktop/CO2_Emission/trained_model.sav', 'rb'))

# Define mappings for dropdown menus
vehicle_class_mapping = {
    'Compact': 0, 'Full-size': 1, 'Mid-size': 2, 'Minicompact': 3, 'Minivan': 4,
    'Pickup truck: Small': 5, 'Pickup truck: Standard': 6, 'SUV: Small': 7, 'SUV: Standard': 8,
    'Special purpose vehicle': 9, 'Station wagon: Mid-size': 10, 'Station wagon: Small': 11,
    'Subcompact': 12, 'Two-seater': 13, 'Van: Passenger': 14
}

transmission_mapping = {
    'A10': 0, 'A6': 1, 'A8': 2, 'A9': 3, 'AM6': 4, 'AM7': 5, 'AM8': 6, 'AM9': 7, 'AS10': 8,
    'AS5': 9, 'AS6': 10, 'AS7': 11, 'AS8': 12, 'AS9': 13, 'AV': 14, 'AV1': 15, 'AV10': 16,
    'AV6': 17, 'AV7': 18, 'AV8': 19, 'M5': 20, 'M6': 21, 'M7': 22
}

model_grouped_mapping = {
    'Camaro': 0, 'Canyon 4WD': 1, 'Challenger': 2, 'Cherokee': 3, 'Civic Sedan': 4,
    'Corolla': 5, 'Elantra': 6, 'F-150 FFV': 7, 'F-150 FFV 4X4': 8, 'Mustang': 9,
    'Mustang Convertible': 10, 'Other': 11, 'Sierra': 12, 'Sierra 4WD': 13,
    'Sierra 4WD AT4': 14, 'Silverado': 15, 'Silverado 4WD': 16, 
    'Silverado 4WD Trail Boss': 17, 'Suburban 4WD': 18, 'Yukon 4WD': 19, 
    'Yukon XL 4WD': 20
}

make_grouped_mapping = {
    'Audi': 0, 'BMW': 1, 'Cadillac': 2, 'Chevrolet': 3, 'Dodge': 4, 'Ford': 5, 
    'GMC': 6, 'Honda': 7, 'Hyundai': 8, 'Jeep': 9, 'Kia': 10, 'Lexus': 11, 'MINI': 12, 
    'Mazda': 13, 'Mercedes-Benz': 14, 'Nissan': 15, 'Other': 16, 'Porsche': 17, 
    'Subaru': 18, 'Toyota': 19, 'Volkswagen': 20
}

def emission(input_data):
    # Convert input to NumPy array
    input_as_array = np.asarray(input_data)

    # Reshape the input array for prediction
    input_reshaped = input_as_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_reshaped)
    prediction = round(float(prediction[0]), 2)

    return f'Predicted CO2 Emissions: {prediction}g/km'


def main():
    st.title('CO2 Emissions Prediction')
    st.image(r'C:\Users\Dell\Desktop\CO2_Emission\Veering_Off_Course.jpg', use_column_width=True)


    # Numerical inputs
    engine_size_log = st.number_input('Engine Size (Log):')
    cylinders_sqrt = st.number_input('Cylinders (Square Root):')
    fuel_city_log = st.number_input('Fuel Consumption in City (Log):')
    fuel_highway_log = st.number_input('Fuel Consumption on Highway (Log):')
    fuel_combined = st.number_input('Fuel Consumption Combined (L/100km):')
    smog_level_log = st.number_input('Smog Level (Log):')
    
    # Dropdowns for categorical features
    vehicle_class = st.selectbox('Vehicle Class:', list(vehicle_class_mapping.keys()))
    transmission = st.selectbox('Transmission:', list(transmission_mapping.keys()))
    model_grouped = st.selectbox('Model:', list(model_grouped_mapping.keys()))
    make_grouped = st.selectbox('Manufacturer:', list(make_grouped_mapping.keys()))
    
    # Convert selected names to encoded values
    vehicle_class_encoded = vehicle_class_mapping[vehicle_class]
    transmission_encoded = transmission_mapping[transmission]
    model_grouped_encoded = model_grouped_mapping[model_grouped]
    make_grouped_encoded = make_grouped_mapping[make_grouped]

    # Code for prediction
    pred = ''
    
    # Button for prediction
    if st.button('Predict Your Result'):
        pred = emission([engine_size_log, cylinders_sqrt, fuel_city_log, fuel_highway_log,
                         fuel_combined, smog_level_log, vehicle_class_encoded,
                         transmission_encoded, model_grouped_encoded, make_grouped_encoded])
        
    st.success(pred)
    

if __name__ == '__main__':
    main()
