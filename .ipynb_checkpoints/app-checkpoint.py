import streamlit as st
import joblib
import pandas as pd

# Load the model and feature order
model, feature_order = joblib.load('kidney_disease_model.pkl')

# Title and introduction
st.title('Chronic Kidney Disease Prediction')
st.write('This app predicts whether a patient has chronic kidney disease based on their medical data.')

# User inputs
age = st.slider('Age', 20, 80, value=30)
bp = st.slider('Blood Pressure', 70, 180, value=120)
sg = st.slider('Specific Gravity', 1.005, 1.030, value=1.020)
al = st.slider('Albumin', 0, 5, value=0)
su = st.slider('Sugar', 0, 5, value=0)
bgr = st.slider('Blood Glucose Random', 70, 200, value=90)
bu = st.slider('Blood Urea', 10, 50, value=15)
sc = st.slider('Serum Creatinine', 0.6, 1.5, value=1.0)
sod = st.slider('Sodium', 135, 145, value=140)
pot = st.slider('Potassium', 3.5, 5.5, value=4.5)
hemo = st.slider('Hemoglobin', 10.0, 17.5, value=15.0)
pcv = st.slider('Packed Cell Volume', 30, 50, value=45)
wbcc = st.slider('White Blood Cell Count', 4000, 11000, value=6000)
rbcc = st.slider('Red Blood Cell Count', 3.5, 5.5, value=4.5)
htn_yes = st.selectbox('Hypertension', [0, 1])
dm_yes = st.selectbox('Diabetes Mellitus', [0, 1])
cad_yes = st.selectbox('Coronary Artery Disease', [0, 1])
appet_poor = st.selectbox('Appetite', [0, 1])
pe_yes = st.selectbox('Pedal Edema', [0, 1])
ane_yes = st.selectbox('Anemia', [0, 1])

# Initialize all features with default values (e.g., 0 for binary features)
input_data = pd.DataFrame(columns=feature_order)
input_data.loc[0] = 0  # Initialize with zeroes

# Assign the user inputs to the appropriate features
input_data['age'] = age
input_data['bp'] = bp
input_data['sg'] = sg
input_data['al'] = al
input_data['su'] = su
input_data['bgr'] = bgr
input_data['bu'] = bu
input_data['sc'] = sc
input_data['sod'] = sod
input_data['pot'] = pot
input_data['hemo'] = hemo
input_data['pcv'] = pcv
input_data['wbcc'] = wbcc
input_data['rbcc'] = rbcc
input_data['htn_yes'] = htn_yes
input_data['dm_yes'] = dm_yes
input_data['cad_yes'] = cad_yes
input_data['appet_poor'] = appet_poor
input_data['pe_yes'] = pe_yes
input_data['ane_yes'] = ane_yes

# Add interaction terms
input_data['bp_x_sc'] = bp * sc
input_data['bu_x_sc'] = bu * sc
input_data['hemo_x_pcv'] = hemo * pcv

# Ensure the input data columns are in the correct order
input_data = input_data.reindex(columns=feature_order)

# Prediction button
if st.button('Predict'):
    # Obtain probability predictions
    y_prob = model.predict_proba(input_data)
    st.write(f'Probabilities: {y_prob}')

    # Set a custom threshold for classification
    threshold = 0.6  # Adjust the threshold as needed

    # Convert probabilities to binary predictions based on the threshold
    prediction = (y_prob[:, 1] >= threshold).astype(int)

    # Display the predicted class and probabilities
    st.write(f'The predicted class is: {"ckd" if prediction[0] else "notckd"}')
    st.write(f'Probability of CKD: {y_prob[0][1]:.2f}')
    st.write(f'Probability of not CKD: {y_prob[0][0]:.2f}')
