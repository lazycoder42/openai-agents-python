import pandas as pd

# Load the CSV file
file_path = 'API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_19358.csv'
df = pd.read_csv(file_path, skiprows=4)

# Filter for the United Kingdom
uk_data = df[df['Country Name'] == 'United Kingdom']

# Select the last 10 years of GDP data
year_columns = [col for col in df.columns if col.isdigit() or col.startswith('20') or col.startswith('19')]
last_10_years = year_columns[-10:]
result = uk_data.loc[:, ['Country Name'] + last_10_years]

# Drop rows with NaN values
result = result.dropna()

result