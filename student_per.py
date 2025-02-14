import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl", "rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocessing_input_data(data,scaler, le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance Prediction App")
    st.write("enter your data to get a prediction for your performance")
    hours_studied= st.number_input("hours studied", min_value=1, max_value=10, value=5)
    Previous_Scores= st.number_input("Previous Scores", min_value=40, max_value=100, value=75)
    Extracurricular_Activities= st.selectbox("Extracurricular Activities", ["Yes", "No"])
    Sleeping_Hours= st.number_input("Sleeping Hours", min_value=5, max_value=10, value=7)
    Sample_Question_Papers_Practiced = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)

    if st.button("Predict your performance"):
        user_data = {"Hours Studied": hours_studied,
            "Previous Scores" :Previous_Scores,
            "Extracurricular Activities": Extracurricular_Activities,
            "Sleep Hours" : Sleeping_Hours,
            "Sample Question Papers Practiced" : Sample_Question_Papers_Practiced }
        
        prediction = predict_data(user_data)
        st.success(f"your prediction score is {prediction}")




if __name__ == "__main__":
    main()

