import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns  # Import Seaborn for visualizations

# Application Title
st.title("Reward Pool Depletion Simulation")

# Creating Tabs
tabs = st.tabs(["Simulation Parameters", "Results Analysis", "Depletion Simulation"])

# Onglet 1 : Simulation Parameters
with tabs[0]:
    st.header("Simulation Parameters")

    # Input ARPU for power users and normal users
    arpu_power_user = st.slider("Power Users ARPU ($)", min_value=0.01, max_value=100.0, value=10.0, step=0.1)
    arpu_normal_user = st.slider("Normal Users ARPU ($)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

    # Input initial number of users on day 0
    initial_users = st.number_input("Initial Number of Users (Day 0)", min_value=0, value=1000, step=100)

    # Input the Shape parameter of the Log-Normal distribution (limited to prevent extreme growth)
    shape_param = st.slider("Shape Parameter of Log-Normal Distribution", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    # Facteur de croissance contrôlé
    growth_factor = st.slider("Facteur de Croissance (% par mois)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

    # Simulation duration parameter
    simulation_months = st.slider("Simulation Duration (in months)", min_value=1, max_value=60, value=48)

    # Button to run the simulation
    if st.button("Run Simulation"):
        simulated_days = simulation_months * 30

        # Simulation with adjustable Log-Normal distribution and growth factor
        shape, loc, scale = shape_param, 0, initial_users * (1 + growth_factor / 100)
        simulated_dau = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=simulated_days)
        simulated_dau = np.clip(simulated_dau, 0, initial_users * 100)  # Limiter la croissance extrême

        power_users = simulated_dau * 0.05
        normal_users = simulated_dau * 0.95
        daily_revenue = (power_users * arpu_power_user) + (normal_users * arpu_normal_user)
        total_revenue = np.sum(daily_revenue)

        st.session_state['simulated_dau'] = simulated_dau
        st.session_state['daily_revenue'] = daily_revenue
        st.session_state['total_revenue'] = total_revenue
        st.success("Simulation completed successfully!")

# Onglet 2 : Results Analysis
with tabs[1]:
    st.header("Results Analysis")

    if 'simulated_dau' in st.session_state:
        simulated_dau = st.session_state['simulated_dau']
        daily_revenue = st.session_state['daily_revenue']
        total_revenue = st.session_state['total_revenue']

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(simulated_dau, kde=True, color='skyblue')
        ax.set_xlabel('Daily Active Users')
        ax.set_ylabel('Densité de Probabilité')
        ax.set_title("Probability Density of Daily Active Users")
        st.pyplot(fig)

        st.subheader("Simulated Daily Revenue ($)")
        st.line_chart(daily_revenue)

        st.metric("Total Simulated Revenue ($)", f"{total_revenue:,.2f}")
        st.metric("Total Active Users", f"{int(np.sum(simulated_dau)):,}")
    else:
        st.warning("Please run the simulation in the previous tab first.")

# Onglet 3 : Depletion Simulation
with tabs[2]:
    st.header("Simulation de la Déplétion de la Pool de Récompenses")

    token_price = st.number_input("Linear Token Price ($)", min_value=0.01, value=0.10, step=0.01)
    reward_pool_tokens = st.number_input("Total Tokens Allocated to the Reward Pool", min_value=1000, value=100000, step=1000)
    redistribution_percentage = st.slider("% of Revenue Redistributed to Stakers", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    if 'daily_revenue' in st.session_state:
        daily_revenue = st.session_state['daily_revenue']
        daily_tokens_distributed = (daily_revenue * (redistribution_percentage / 100)) / token_price
        cumulative_tokens_distributed = np.cumsum(daily_tokens_distributed)
        tokens_remaining = reward_pool_tokens - cumulative_tokens_distributed
        tokens_remaining = np.clip(tokens_remaining, 0, None)

        days_until_depletion = np.argmax(cumulative_tokens_distributed >= reward_pool_tokens) + 1
        if days_until_depletion == 1 and cumulative_tokens_distributed[0] < reward_pool_tokens:
            days_until_depletion = len(cumulative_tokens_distributed)

        st.subheader("Simulation de la Déplétion de la Pool")
        st.metric("Number of Days Until Depletion", f"{days_until_depletion} jours")

        # Graph of remaining tokens vs token price
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(tokens_remaining, label='Tokens Restants dans la Pool', color='green')
        ax1.set_xlabel('Jours')
        ax1.set_ylabel('Tokens Restants', color='green')
        ax1.tick_params(axis='y', labelcolor='green')

        ax2 = ax1.twinx()
        ax2.plot([token_price] * len(tokens_remaining), label='Prix du Token', color='blue', linestyle='--')
        ax2.set_ylabel('Prix du Token ($)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title("Remaining Tokens in the Pool vs Token Price")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Veuillez d'abord lancer la simulation dans l'onglet 'Simulation Parameters'.")
