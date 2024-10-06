import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
with open(r"C:/Users/anant/OneDrive/Desktop/jupyter/model_best2.pkl", 'rb') as file:
    model = pickle.load(file)

# Load the original data for user-friendly display
original_data = pd.read_csv(r"C:/Users/anant/OneDrive/Desktop/jupyter/use_original.csv")

# Initialize label encoders and scaler
label_encoders = {}
categorical_columns = ['city', 'model', 'variantName', 'Color', 'Engine Type', 'Fuel Suppy System', 'Gear Box', 'RTO']
scaler = StandardScaler()

# Fit label encoders on original data to prepare for input transformation
for column in categorical_columns:
    le = LabelEncoder()
    original_data[column] = original_data[column].astype(str)  # Ensure the column is of string type
    le.fit(original_data[column])
    label_encoders[column] = le

# Fit the scaler on the numerical features from the original data (or training data)
numerical_features = ['Width', 'Max Power', 'km', 'RPM', 'price_per_mileage', 'Mileage', 'Height', 'car_age', 'Year of Manufacture']
scaler.fit(original_data[numerical_features])  # Fit on your original or training data

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the car features below and get an estimated price.")

# User selects city, and filter RTO based on the selected city
cities = original_data['city'].unique().tolist()
selected_city = st.selectbox("City", cities)

# Filter the RTO options based on the selected city
filtered_rto = original_data[original_data['city'] == selected_city]['RTO'].unique().tolist()
rto = st.selectbox("RTO", filtered_rto)

# Categorical inputs for other features (not filtered by city)
car_models = original_data['model'].unique().tolist()
variant_names = original_data['variantName'].unique().tolist()
colors = original_data['Color'].unique().tolist()
engine_types = original_data['Engine Type'].unique().tolist()
fuel_supply_systems = original_data['Fuel Suppy System'].unique().tolist()  # Corrected here
gear_boxes = original_data['Gear Box'].unique().tolist()

car_model = st.selectbox("Car Model", car_models)
variant_name = st.selectbox("Variant Name", variant_names)
color = st.selectbox("Color", colors)
engine_type = st.selectbox("Engine Type", engine_types)
fuel_supply_system = st.selectbox("Fuel Suppy System", fuel_supply_systems)  # Corrected here
gear_box = st.selectbox("Gear Box", gear_boxes)

# Numerical inputs
manufacture_year = st.selectbox("Year of Manufacture", original_data['Year of Manufacture'].unique().tolist())
width = st.number_input("Width(mm)", min_value=1000, max_value=3000, value=2000)
max_power = st.number_input("Max Power (hp)", min_value=0, max_value=500, value=100)
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=20000)
rpm = st.number_input("RPM", min_value=0, max_value=10000, value=3000)
price_per_mileage = st.number_input("Price Per Mileage (₹/kmpl)", min_value=0.0, max_value=100.0, value=15.0)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=10.0)
height = st.number_input("Height(mm)", min_value=1000, max_value=3000, value=1500)
car_age = st.slider("Car Age (years)", min_value=0, max_value=20, value=5)

# Button to predict price
if st.button('Predict Price'):
    # Create a new DataFrame with the input features
    new_data = pd.DataFrame({
        'Year of Manufacture': [manufacture_year],
        'Width': [width],
        'Max Power': [max_power],
        'km': [kilometers_driven],
        'RPM': [rpm],
        'price_per_mileage': [price_per_mileage],
        'Mileage': [mileage],
        'Height': [height],
        'car_age': [car_age],
        'city': [selected_city],
        'model': [car_model],
        'variantName': [variant_name],
        'Color': [color],
        'Engine Type': [engine_type],
        'Fuel Suppy System': [fuel_supply_system],  # Corrected here
        'Gear Box': [gear_box],
        'RTO': [rto]
    })

    # Prepare the input for prediction
    encoded_data = new_data.copy()

    # Map categorical columns to their numeric codes using the fitted label encoders
    for column in label_encoders:
        encoded_data[column] = label_encoders[column].transform(encoded_data[column])

    # Scale numerical features (using the fitted StandardScaler)
    encoded_data[numerical_features] = scaler.transform(encoded_data[numerical_features])

    # Ensure features are in the same order as during training
    model_features = ['RTO', 'Engine Type', 'city', 'Gear Box', 'Fuel Suppy System', 'Color', 'variantName', 'model', 'price_per_mileage', 'Mileage', 'km', 'RPM', 'car_age', 'Max Power', 'Year of Manufacture', 'Height', 'Width']

    # Rearranging the encoded_data DataFrame to match the model's training order
    encoded_data = encoded_data[model_features]

    # Make the prediction
    predicted_price = model.predict(encoded_data)

 

    # Format the predicted price to ensure it displays as required
    predicted_price_value = predicted_price[0]
  

    # Display the price correctly
    st.success(f"The predicted price of the car is: ₹ {predicted_price_value} Lakhs")  # Display in lakhs