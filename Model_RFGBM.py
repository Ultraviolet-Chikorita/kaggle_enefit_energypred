import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt

chunk_size = 100000  # Adjust this based on your available memory
chunks = pd.read_csv("E://Kaggle/Power/RFGBM/Train/train.csv", chunksize=chunk_size)

model = SGDRegressor(random_state=42)
scaler = StandardScaler()
label_encoder = LabelEncoder()
weather_df = pd.read_csv("E://Kaggle/Power/RFGBM/Train/historical_weather.csv")
weather_county_df = pd.read_csv("E://Kaggle/Power/RFGBM/Train/weather_station_to_county_mapping.csv")
weather_df = weather_df.merge(weather_county_df)
electricity_prices_df = pd.read_csv("E://Kaggle/Power/RFGBM/Train/electricity_prices.csv")
rmse_list = []
for chunk in chunks:
    # Merge with other dataframes
    chunk = chunk.merge(electricity_prices_df, on='data_block_id')

    # Preprocess chunk...

    # Convert datetime, label encode, extract features, etc.
    chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    chunk['hour'] = chunk['datetime'].dt.hour
    chunk['day'] = chunk['datetime'].dt.day
    chunk['month'] = chunk['datetime'].dt.month
    
    chunk['county_encoded'] = label_encoder.fit_transform(chunk['county'])
    chunk['product_type_encoded'] = label_encoder.fit_transform(chunk['product_type'])
    chunk = chunk.dropna(subset=['target'])
    X_chunk = chunk[['county_encoded', 'is_business', 'product_type_encoded', 'hour', 'day', 'month', 'euros_per_mwh']]
    y_chunk = chunk['target']
    if X_chunk.empty:
        print("X_chunk is empty before scaling. Skipping this chunk.")
        continue
    
    # Scale features
    X_chunk = scaler.fit_transform(X_chunk)
    
    # Split chunk into train and validation
    X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)
    # Partially fit the model
    model.partial_fit(X_train_chunk, y_train_chunk)
    y_pred = model.predict(X_val_chunk)
    rmse = mean_squared_error(y_val_chunk, y_pred, squared=False)
    print(f'Validation RMSE: {rmse}')
    rmse_list.append(rmse)
plt.plot(rmse_list)
plt.show()
# Predict and evaluate
# After all chunks are processed, evaluate the model

# Make predictions on the test set (assuming you have a test set ready)
# test_df = ...
# test_predictions = model.predict(test_df)
# Prepare submission file
# submission_df = ...
# submission_df.to_csv('submission.csv', index=False)

