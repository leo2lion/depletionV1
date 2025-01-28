import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import zlib

# Application Title
st.title("Reward Pool Depletion Simulation")

# Creating Tabs
tabs = st.tabs(["Simulation Parameters", "Results Analysis", "Depletion Simulation"])

# Onglet 1 : Simulation Parameters
with tabs[0]:
    st.header("Simulation Parameters")

    arpu_power_user = st.slider("Power Users ARPU ($)", 0.01, 100.0, 10.0, 0.1)
    arpu_normal_user = st.slider("Normal Users ARPU ($)", 0.01, 10.0, 1.0, 0.1)
    initial_users = st.number_input("Initial Number of Users (Day 0)", 0, 100000, 1000, 100)
    shape_param = st.slider("Shape Parameter of Log-Normal Distribution", 0.1, 2.0, 0.5, 0.1)
    growth_factor = st.slider("Growth Factor (% per month)", 0.0, 50.0, 5.0, 0.5)
    simulation_months = st.slider("Simulation Duration (months)", 1, 60, 48)

    if st.button("Run Simulation"):
        simulated_days = simulation_months * 30
        shape, loc, scale = shape_param, 0, initial_users * (1 + growth_factor / 100)
        simulated_dau = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=simulated_days)
        simulated_dau = np.clip(simulated_dau, 0, initial_users * 100)
        
        power_users = simulated_dau * 0.05
        normal_users = simulated_dau * 0.95
        daily_revenue = (power_users * arpu_power_user) + (normal_users * arpu_normal_user)
        
        st.session_state['simulated_dau'] = simulated_dau
        st.session_state['daily_revenue'] = daily_revenue
        st.session_state['total_revenue'] = np.sum(daily_revenue)
        st.success("Simulation completed successfully!")

# Onglet 2 : Results Analysis
with tabs[1]:
    st.header("Results Analysis")

    if 'simulated_dau' in st.session_state:
        dau = st.session_state['simulated_dau']
        revenue = st.session_state['daily_revenue']
        total = st.session_state['total_revenue']

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.histplot(dau, kde=True, color='skyblue')
        ax1.set(xlabel='Daily Active Users', ylabel='Density', title='User Distribution')
        st.pyplot(fig1)

        st.subheader("Daily Revenue Evolution")
        st.line_chart(revenue)

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue ($)", f"{total:,.2f}")
        col2.metric("Total User-Days", f"{int(np.sum(dau)):,}")
    else:
        st.warning("Please run the simulation first in the Parameters tab.")

# Onglet 3 : Depletion Simulation
with tabs[2]:
    st.header("Reward Pool Depletion Analysis")
    
    price_model = st.radio("Token Pricing Model", ["Fixed Price", "Stochastic (GBM)"])
    
    if price_model == "Fixed Price":
        token_price = st.number_input("Token Price ($)", 0.01, 1000.0, 0.10, 0.01)
    else:
        start_price = st.number_input("Initial Token Price ($)", 0.01, 1000.0, 0.10, 0.01)
        drift = st.slider("Drift (%)", -10.0, 10.0, 0.0, 0.1)
        volatility = st.slider("Volatility (%)", 0.0, 100.0, 20.0, 0.1)
        n_simulations = st.slider("Number of Paths", 100, 10000, 1000)

    reward_pool = st.number_input("Reward Pool Size (tokens)", 1000, 10000000, 100000, 1000)
    redistribution = st.slider("Revenue Redistribution (%)", 0.0, 100.0, 20.0, 1.0)

    if 'daily_revenue' in st.session_state:
        daily_revenue = st.session_state['daily_revenue']
        days = len(daily_revenue)
        
        if price_model == "Stochastic (GBM)":
            # Generate reproducible seed from parameters
            params_hash = zlib.crc32(f"{drift}_{volatility}".encode()) & 0xffffffff
            np.random.seed(params_hash)
            
            # Monte Carlo simulations
            dt = 1/252
            paths = []
            for _ in range(n_simulations):
                prices = [start_price]
                for __ in range(days-1):
                    shock = np.random.normal()
                    new_price = prices[-1] * np.exp((drift/100 - 0.5*(volatility/100)**2)*dt + 
                                                   (volatility/100)*np.sqrt(dt)*shock)
                    prices.append(new_price)
                paths.append(prices)
            
            # Find percentile representative paths
            final_prices = np.array([p[-1] for p in paths])
            p5_val = np.percentile(final_prices, 5)
            p95_val = np.percentile(final_prices, 95)
            
            p5_idx = np.argmin(np.abs(final_prices - p5_val))
            median_idx = np.argsort(final_prices)[len(final_prices)//2]
            p95_idx = np.argmin(np.abs(final_prices - p95_val))
            
            # Store in session state
            st.session_state['paths'] = {
                "P5 (Conservative)": paths[p5_idx],
                "Median (Expected)": paths[median_idx],
                "P95 (Optimistic)": paths[p95_idx]
            }
            
            # Scenario selection
            scenario = st.selectbox("Select Price Scenario", list(st.session_state['paths'].keys()))
            selected_path = st.session_state['paths'][scenario]
            
            # Calculate depletion
            daily_tokens = (daily_revenue * (redistribution/100)) / selected_path
            cumulative_tokens = np.cumsum(daily_tokens)
            remaining_tokens = np.maximum(reward_pool - cumulative_tokens, 0)
            
            # Find depletion day
            depletion_day = np.argmax(cumulative_tokens >= reward_pool) + 1
            if depletion_day == 1 and cumulative_tokens[0] < reward_pool:
                depletion_day = days
            
            # Visualization
            fig2, ax = plt.subplots(figsize=(12,6))
            ax.plot(remaining_tokens, color='darkred', label='Remaining Tokens')
            ax.set(xlabel='Days', ylabel='Tokens Remaining', title=f'Depletion Timeline - {scenario}')
            ax.grid(True)
            
            ax2 = ax.twinx()
            for label, path in st.session_state['paths'].items():
                ax2.plot(path, linestyle='--' if "P5" in label else '-.', alpha=0.7, label=label)
            ax2.set_ylabel('Token Price ($)')
            
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper center')
            
            st.pyplot(fig2)
            st.metric("Days Until Depletion", f"{depletion_day} days")
        
        else:  # Fixed price case
            token_prices = np.full(days, token_price)
            daily_tokens = (daily_revenue * (redistribution/100)) / token_prices
            cumulative_tokens = np.cumsum(daily_tokens)
            remaining_tokens = np.maximum(reward_pool - cumulative_tokens, 0)
            
            depletion_day = np.argmax(cumulative_tokens >= reward_pool) + 1
            if depletion_day == 1 and cumulative_tokens[0] < reward_pool:
                depletion_day = days
            
            fig2, ax = plt.subplots(figsize=(12,6))
            ax.plot(remaining_tokens, color='darkred', label='Tokens Remaining')
            ax.set(xlabel='Days', ylabel='Tokens', title='Fixed Price Depletion')
            ax.grid(True)
            
            ax2 = ax.twinx()
            ax2.plot(token_prices, color='blue', linestyle=':', label='Token Price')
            ax2.set_ylabel('Price ($)')
            
            ax.legend(loc='upper left')
            st.pyplot(fig2)
            st.metric("Days Until Depletion", f"{depletion_day} days")
            
    else:
        st.warning("Please run the simulation in the Parameters tab first.")
