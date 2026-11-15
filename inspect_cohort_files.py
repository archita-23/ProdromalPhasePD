import pandas as pd
from pathlib import Path

files = [
    Path(r"C:\Users\HP\OneDrive\Desktop\PBL before preprocessing\authentic_data.csv"),
    Path(r"C:\Users\HP\OneDrive\Desktop\PBL before preprocessing\training_dataset.csv"),
    Path(r"C:\Users\HP\OneDrive\Desktop\PBL_old\prodromal_data.csv"),
    Path(r"C:\Users\HP\OneDrive\Desktop\PPMI COMPRESSED.xlsx"),
]

for file in files:
    print("\n" + "=" * 70)
    print("FILE:", file.name)

    try:
        if file.suffix.lower() == ".csv":
            df = pd.read_csv(file, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.head(3).to_string(index=False))

        elif file.suffix.lower() == ".xlsx":
            excel = pd.ExcelFile(file)
            print("Sheets:", excel.sheet_names)

            for sheet in excel.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet, nrows=3)
                print("\nSheet:", sheet)
                print("Columns:", df.columns.tolist())
                print(df.head(2).to_string(index=False))

    except Exception as e:
        print("Error:", e)