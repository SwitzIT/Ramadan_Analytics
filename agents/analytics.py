import pandas as pd

class AnalyticsAgent:
    def __init__(self):
        self.pre_ramadan_start = "2026-01-18"
        self.pre_ramadan_end = "2026-02-17"
        self.ramadan_start = "2026-02-18"
        self.ramadan_end = "2026-03-20"

    def analyze_performance(self, df):
        """Analyzes performance comparison between Pre-Ramadan and Ramadan."""
        # Ensure date format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter data for periods
        pre_df = df[(df['Date'] >= self.pre_ramadan_start) & (df['Date'] <= self.pre_ramadan_end)]
        ram_df = df[(df['Date'] >= self.ramadan_start) & (df['Date'] <= self.ramadan_end)]
        
        # Aggregate Net Sales per Shop
        pre_sales = pre_df.groupby(['Sold-To Party', 'Name'])['Adjusted Value'].sum().reset_index()
        pre_sales.rename(columns={'Adjusted Value': 'Pre-Ramadan Net Sales'}, inplace=True)
        
        ram_sales = ram_df.groupby(['Sold-To Party', 'Name'])['Adjusted Value'].sum().reset_index()
        ram_sales.rename(columns={'Adjusted Value': 'Ramadan Net Sales'}, inplace=True)
        
        # Merge results
        final_df = pd.merge(pre_sales, ram_sales, on=['Sold-To Party', 'Name'], how='outer').fillna(0)
        
        # Calculate Drop %
        # Avoid division by zero
        final_df['Drop %'] = ((final_df['Pre-Ramadan Net Sales'] - final_df['Ramadan Net Sales']) / 
                              final_df['Pre-Ramadan Net Sales'].replace(0, float('inf'))) * 100
        
        # Handle cases where Pre-Ramadan was 0 but Ramadan had sales (negative drop)
        final_df.loc[final_df['Pre-Ramadan Net Sales'] == 0, 'Drop %'] = 0
        
        # Classify drops
        def classify_drop(drop_pct):
            if drop_pct < 20: return "Stable"
            elif 20 <= drop_pct < 40: return "Moderate Drop"
            elif 40 <= drop_pct < 60: return "High Drop"
            else: return "Critical Drop"
            
        final_df['Classification'] = final_df['Drop %'].apply(classify_drop)
        
        return final_df
