import streamlit as st 
from joblib import load
import warnings
import numpy as np
import pandas as pd

st.title("Iris Classification")

#loading model andd scaler
try:
    model = load('./model/model.pkl')
    scaler = load('./model/scaler.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# form to handle user input
with st.form(key='my_form'):
    col1, col2 = st.columns(2)
    # each input field has predefined value 
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.2, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=2.7, step=0.1)
    
    with col2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.9, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)

    submit_button = st.form_submit_button(label='Predict Species')

# submissions when user will click submit button
if submit_button:
    # so we have to make sure that user entered all the measurements 
    if sepal_length and sepal_width and petal_length and petal_width:
        try:
            #creating input data from user input values and scale it
            input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                    columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
            input_scaled = scaler.transform(input_data)
            
            # getting the predictions and the probabilites
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # mapping the predictions number value to specieis name
            species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
            predicted_species = species_mapping[prediction[0]]
            

            # [[[[[RESULTS]]]]]

            st.markdown("### Prediction Results")
            
            #three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Species", predicted_species)
            
            with col2:
                confidence = prediction_proba[prediction[0]] * 100
                st.metric("Prediction Confidence", f"{confidence:.1f}%")
            
            with col3:
                #calculating reliability based on difference between top two probabilities
                sorted_proba = sorted(prediction_proba, reverse=True)
                reliability = (sorted_proba[0] - sorted_proba[1]) * 100  # Difference between top two predictions
                st.metric("Model Reliability", f"{reliability:.1f}%")
            
            # probability distributions
            st.markdown("### Probability Distribution for Each Species")
            prob_df = pd.DataFrame({
                'Species': ['Setosa', 'Versicolor', 'Virginica'],
                'Probability': prediction_proba * 100
            })
            
            # plotting the bar chart for probabilities
            st.bar_chart(prob_df.set_index('Species'))
           

            st.markdown("### Interpretation")
            st.markdown(f"""
            - Predicted Species: **{predicted_species}**
            - Confidence: **{confidence:.1f}%** (probability of the predicted species)
            - Reliability: **{reliability:.1f}%** (distinction from next most likely species)
            """)

            #reliability
            if reliability > 50:
                st.success("High reliability: Clear distinction between species!")
            elif reliability > 20:
                st.info("Moderate reliability: Some overlap between species possible.")
            else:
                st.warning("Low reliability: Close call between different species.")

            
        except Exception as e:
            st.error(f"error making prediction: {e}")
    else:
        st.error("Please fill in all fields with valid measurements.") 

# Non-functional : have to add a short about section
with st.expander("About the Model"):
    st.markdown("""
    This model was trained on the famous Iris dataset and uses machine learning to predict the species of Iris flowers.
    The prediction is based on four measurements:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
    
    The model provides two key metrics:
    - Confidence: How likely the predicted species is
    - Reliability: How distinct the prediction is from other species
    
    If You Have Any Query Please Write <a href="mailto:bc200415701@vu.edu.pk">bc200415701@vu.edu.pk</a> 
    """, unsafe_allow_html=True)