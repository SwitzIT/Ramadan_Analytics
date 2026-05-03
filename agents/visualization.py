import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

class VisualizationAgent:
    def _format_inr(self, number):
        """Formats a number into Indian Rupee format (e.g., 1,00,000)"""
        is_negative = number < 0
        number = abs(int(number))
        s = str(number)
        if len(s) <= 3:
            res = s
        else:
            res = s[-3:]
            s = s[:-3]
            while len(s) > 2:
                res = s[-2:] + "," + res
                s = s[:-2]
            res = s + "," + res
        return f"-₹{res}" if is_negative else f"₹{res}"

    def render_kpis(self, analysis_df):
        total_pre = analysis_df['Pre-Ramadan Net Sales'].sum()
        total_ram = analysis_df['Ramadan Net Sales'].sum()
        avg_drop = analysis_df['Drop %'].mean()
        total_stores = len(analysis_df)
        
        # Row 1: High level
        st.subheader("Financial Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stores", f"{total_stores:,}")
        col2.metric("Pre-Ramadan Total", self._format_inr(total_pre))
        col3.metric("Ramadan Total", self._format_inr(total_ram))
        col4.metric("Avg Drop %", f"{avg_drop:.1f}%")
        
        st.markdown("---")
        
        # Row 2: Categorization
        st.subheader("Store Performance Breakdown")
        cat_counts = analysis_df['Classification'].value_counts()
        stable = cat_counts.get('Stable', 0)
        moderate = cat_counts.get('Moderate Drop', 0)
        high = cat_counts.get('High Drop', 0)
        critical = cat_counts.get('Critical Drop', 0)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stable (<20% drop)", f"{stable:,}")
        c2.metric("Moderate Drop (20-40%)", f"{moderate:,}")
        c3.metric("High Drop (40-60%)", f"{high:,}")
        c4.metric("Critical Drop (>60%)", f"{critical:,}")

    def render_charts(self, analysis_df, full_df):
        # Top 10 Declining Shops
        st.subheader("Top 10 Declining Shops (by Drop %)")
        top_drops = analysis_df.sort_values(by='Drop %', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top_drops['Name'], top_drops['Drop %'], color='salmon')
        ax.set_ylabel("Drop %")
        ax.set_title("Highest Percentage Sales Drop")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        # Sales Trend
        st.subheader("Sales Trend (Daily Adjusted Value)")
        daily_sales = full_df.groupby('Date')['Adjusted Value'].sum().reset_index()
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(daily_sales['Date'], daily_sales['Adjusted Value'], color='teal', marker='o', markersize=4)
        ax2.set_ylabel("Net Sales")
        ax2.set_xlabel("Date")
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    def render_table(self, analysis_df):
        st.subheader("Detailed Shop Analysis")
        st.dataframe(analysis_df.style.background_gradient(subset=['Drop %'], cmap='Reds'))
