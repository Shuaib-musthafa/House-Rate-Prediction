# House Price Prediction App

This Streamlit-based web application predicts house prices based on various features like the number of bedrooms, bathrooms, living area, lot area, and more. It leverages machine learning to provide users with an estimated price for a house given specific input features.

## Features

- **Predict house prices**: Input features such as the number of bedrooms, bathrooms, living area, lot area, and other details to get an estimated price for a house.
- **Interactive UI**: Built using Streamlit, the app allows for seamless interaction and dynamic updates.
- **Data-driven predictions**: Utilizes a machine learning model trained on historical house price data.

## Installation

To run this app on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/house-price-prediction.git
   cd house-price-prediction
2.Install the required dependencies:
  ```bash
    pip install -r requirements.txt
```
3.Run the Streamlit app:

  ```bash
    streamlit run app.py
```

## Usage
Once the app is running, you can enter the details for the house:

Number of bedrooms
Number of bathrooms
Living area (in square feet)
Lot area (in square feet)
Number of floors
Whether the house has a waterfront
Condition of the house
Year the house was built or renovated
Nearby amenities like schools, airport, etc.
Click on the "Predict" button to get the estimated price for the house.

## Model
The app uses a machine learning model trained on a dataset of historical house prices. The model uses features like the size of the house, number of bedrooms, and proximity to amenities to predict the price.

## Dataset
The dataset contains the following columns:

ID
Date
Number of bedrooms
Number of bathrooms
Living area
Lot area
Number of floors
Waterfront present
Condition of the house
Year built
Renovation year
Postal code
Latitude & Longitude
Living area renovation
Lot area renovation
Number of schools nearby
Distance from the airport
Price (target)
Requirements
Python 3.x
Streamlit
Pandas
Scikit-learn
Matplotlib (optional for visualizations)
##License
This project is licensed under the MIT License. See the LICENSE file for more details.
