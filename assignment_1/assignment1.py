from ucimlrepo import fetch_ucirepo

# fetch dataset
computer_hardware = fetch_ucirepo(id=29)

# data (as pandas dataframes)
dataset = computer_hardware.data.features

# Numerically-measured variables
numerical_variables = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]

# Calculate descriptive measures
descriptive_measures = dataset[numerical_variables].describe().transpose()

# Calculate coefficient of variation
descriptive_measures['Coefficient of Variation'] = descriptive_measures['std'] / descriptive_measures['mean']

# Save the calculated statistics to a spreadsheet
descriptive_measures.to_csv("assignment_1/descriptive_measures.csv")

# Display the calculated statistics
print(descriptive_measures)
