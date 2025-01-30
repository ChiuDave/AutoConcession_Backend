import pandas as pd

def clean_data():
    csv_file = "vehicles.csv"  
    df = pd.read_csv(csv_file)

    df = df[df['SellingPrice'] != 0]
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    clean_data()