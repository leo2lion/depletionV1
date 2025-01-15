# depletionV1
Run your own depletion test to calibrate the amount of tokens in your reward allocation. 

# Depletion Pool Simulation Tool

This Streamlit application is designed to simulate and analyze the depletion of a token reward pool over time. It allows users to model protocol adoption, revenue generation, and token distribution dynamics to predict how long a reward pool can sustain given specific parameters.

## Objectives

The tool is structured into **three main tabs**:

### 1. Simulation Parameters
- Configure the protocol's economic assumptions.
- Simulate the growth of daily active users (DAU) and revenue generation.
- **Inputs:**
  - **Power Users ARPU ($):** Average revenue per power user.
  - **Normal Users ARPU ($):** Average revenue per regular user.
  - **Initial Users:** Number of users at day 0.
  - **Shape Parameter (Log-Normal Distribution):** Controls user growth dispersion.
  - **Growth Factor (% per month):** Adjusts monthly user growth rate.
  - **Simulation Duration (months):** Total time period for the simulation.

### 2. Results Analysis
- Visualize the results of the simulation.
- **Displayed Graphs:**
  - **Probability Density of Daily Active Users:** Distribution of daily active users over time.
  - **Simulated Daily Revenue ($):** Time series of simulated daily revenue.
- **Metrics:**
  - **Total Simulated Revenue ($):** Cumulative revenue over the simulation period.
  - **Total Active Users:** Total number of users throughout the simulation.

### 3. Pool Depletion Simulation
- Simulate how long the reward pool can sustain based on revenue redistribution.
- **Inputs:**
  - **Token Price ($):** Fixed price of the token.
  - **Reward Pool Allocation (Tokens):** Total tokens allocated for rewards.
  - **Revenue Redistribution to Stakers (%):** Percentage of daily revenue redistributed in tokens.
- **Displayed Graphs:**
  - **Remaining Tokens vs. Token Price:** Dual-axis plot showing remaining tokens in the pool and token price over time.
- **Metrics:**
  - **Days Until Depletion:** Estimated number of days before the pool is fully depleted.

## Graph Descriptions

- **Probability Density of Daily Active Users:** Shows how active users are distributed over time, reflecting protocol adoption trends.
- **Simulated Daily Revenue ($):** Represents the protocol's projected daily income.
- **Remaining Tokens vs. Token Price:** Visualizes how quickly the reward pool is consumed based on token distribution and price.

## Running the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/depletion-simulation.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run app.py
   ```

## Dependencies

- Python 3.8+
- Streamlit
- NumPy
- Matplotlib
- SciPy
- Seaborn

## License

This project is licensed under the Creative Commons Legal Code License.

---

For any inquiries or issues, please contact [Leo Delion] at [leo@nomiks.io].

