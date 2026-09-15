import pandas as pd

# Crear DataFrame vacío con la estructura del Weather Dataset
weather_data = {
 'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy'],
 'temperature': ['hot', 'hot', 'hot', 'mild', 'cool'],
 'humidity': ['high', 'high', 'high', 'high', 'normal'],
 'windy': [False, True, False, False, False],
 'play': ['no', 'no', 'yes', 'yes', 'yes']
}

df_weather = pd.DataFrame(weather_data)
print(df_weather.head())