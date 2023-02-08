import pandas as pd
import csv

# Extract Gemeindenamen
file_path = "config/be-t-00.04-agv-01.xlsx"
xl = pd.ExcelFile(file_path)
df = xl.parse(xl.sheet_names[1])
gdename = df.GDENAME.str.lower().tolist()

def list_to_csv(data, file_name):
    with open(file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)

# Example usage
list_to_csv(gdename, 'config/gemeindenamen.csv')

