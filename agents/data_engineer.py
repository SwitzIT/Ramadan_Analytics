import pandas as pd
import os

class DataEngineerAgent:
    def __init__(self, data_path="data/OP.xlsx"):
        self.data_path = data_path

    def load_and_clean_data(self, uploaded_file=None):
        """Loads data from path or uploaded file and cleans it."""
        try:
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
            elif os.path.exists(self.data_path):
                df = pd.read_excel(self.data_path)
            else:
                return None, "Data file not found."

            # Drop unnamed columns often found in Excel exports
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Standardize columns
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Map 'Expairy GR' to 'Expiry GR' if necessary
            df['Type'] = df['Type'].replace({'Expairy GR': 'Expiry GR'})
            
            # Business Logic for Adjusted Value
            # Invoice -> +Value
            # Cancel of CN -> +Value
            # Fresh GR -> -Value
            # Invoice Cancel -> -Value
            # Expiry GR -> -Value
            
            def calculate_adjusted_value(row):
                val = row['MRP Value']
                t = row['Type']
                if t in ['Invoice', 'Cancel of CN']:
                    return val
                elif t in ['Fresh GR', 'Invoice Cancel', 'Expiry GR']:
                    return -val
                return 0

            df['Adjusted Value'] = df.apply(calculate_adjusted_value, axis=1)
            
            return df, None
        except Exception as e:
            return None, str(e)
