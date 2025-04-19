import pandas as pd

# Load the CSV file, skipping metadata/header lines
df = pd.read_csv('API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_19358.csv', skiprows=4)

# Filter the data for the United Kingdom and the last 10 years
result = df[(df['Country Name'] == 'United Kingdom') & (df.columns[-10:].tolist())]

# Select only the relevant columns (last 10 years)
result = result[['Country Name'] + df.columns[-10:].tolist()]

# Display the result
result