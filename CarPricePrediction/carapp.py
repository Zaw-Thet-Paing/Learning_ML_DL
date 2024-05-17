import numpy as np
import streamlit as st
import joblib

# loading out model and scaler
model = joblib.load('xgb_model.pkl')

scaler = joblib.load('scaler.pkl')

def car_price_prediction(input_data):
    # Changing the input into numpy array and reshaping
    input_changed = np.array(input_data).reshape(1,-1)

    # Standardize the input
    std_input = scaler.transform(input_changed)

    prediction = model.predict(std_input)

    return "Estimated car price: " + str(prediction[0])

def main():
    # creating the title
    st.title("Ford car price prediction App")

    # Getting the input from user
    year = st.text_input('Year')
    transmission = st.text_input("Transmission")
    mileage = st.text_input("Mile age")
    fuelType = st.text_input("Fuel Type")
    tax = st.text_input("Tax")
    mpg = st.text_input("MPG")
    engineSize = st.text_input("Engine Size")

    pred_price = ''

    # create a button
    if st.button('Check Estimated Price'):
        pred_price = car_price_prediction([year, transmission, mileage, fuelType, tax, mpg, engineSize])
    
    st.success(pred_price)

if __name__ == "__main__":
    main()