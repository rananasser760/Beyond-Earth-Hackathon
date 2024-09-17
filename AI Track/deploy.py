import streamlit as st
import requests
import joblib
import numpy as np
from streamlit_lottie import st_lottie
from PIL import Image
st.set_page_config(page_title='Smoke Prediction', page_icon = "random")
st.title('Smoke From Fires Prediction and Detection')
def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
def prepare_input_data(UTC,Temperature,Humidity,TVOC,eCO2,Ethanol,Pressure,PM1,PM25,NC05,NC1,NC25,CNT):
    A = [UTC,Temperature,Humidity,TVOC,eCO2,Ethanol,Pressure,PM1,PM25,NC05,NC1,NC25,CNT]
    sample = np.array(A).reshape(-1,len(A))
    return sample

loaded_model = joblib.load(open("logistic_regression_file", 'rb'))



st.write('# Smoke Detection Project')

lottie_link = "https://lottie.host/f021f5f1-e0c0-4d0c-b182-926f133de858/JpRQnfZUNs.json"
animation = load_lottie(lottie_link)

st.write('---')
st.subheader('Enter your details to predict your Loan Status')

with st.container():
    
    right_column, left_column = st.columns(2)
    
    with right_column:
        UTC = st.number_input('UTC (Please write it as day month year minutes hours without space): ')

        Temperature = st.number_input('Temperature: ')
        
        Humidity = st.number_input('Humidity: ')
        
        TVOC = st.number_input('TVOC : ')

        eCO2 = st.number_input('eCO2 : ')
        
        Ethanol = st.number_input('Ethanol : ')
        
        Pressure = st.number_input('Pressure : ')
        
        PM1 = st.number_input('PM 1.0 : ')
        
        PM25 = st.number_input('PM 2.5 : ')
        
        NC05 = st.number_input('NC 0.5 : ')

        NC1 = st.number_input('NC 1.0 : ')

        NC25 = st.number_input('NC 2.5 : ')

        CNT = st.number_input('CNT : ')

        sample = prepare_input_data(UTC,Temperature,Humidity,TVOC,eCO2,Ethanol,Pressure,PM1,PM25,NC05,NC1,NC25,CNT)

    with left_column:
        st_lottie(animation, speed=1, height=400, key="initial")
if st.button('Predict'):
    pred_Y = loaded_model.predict(sample)
    
    if pred_Y == 0:
        st.write('### There is No Fire!')
        st.balloons()
        st.write(pred_Y)
    else:
        st.write('### THERE IS A FIRE!')
        st.write(pred_Y)