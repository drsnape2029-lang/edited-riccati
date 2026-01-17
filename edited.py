import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ================== Riccati ODE ===================
def riccati_rhs(t, y, a):
    return a * y ** 2 + 1 / a

# ================== Streamlit App ===================
def app():
    st.title("Parameterized Riccati Boundary-Value Problem")

    # Sidebar for input parameters
    st.sidebar.header("Settings")
    a = st.sidebar.slider("Parameter (a)", -5.0, 5.0, 1.0, 0.1)
    method = st.sidebar.selectbox("Method", ["Euler", "RK4", "RK45"])
    step_size = st.sidebar.number_input("Step Size", value=0.05)
    tol = st.sidebar.number_input("Tolerance (RK45)", value=1e-6, format="%.1e")
    
    # Time domain
    t0 = 0
    tf = np.pi
    y0 = 0

    # Solve the equation
    t_eval = np.linspace(t0, tf, 500)
    
    if method == "Euler":
        sol = solve_ivp(lambda t, y: riccati_rhs(t, y, a), (t0, tf), [y0], t_eval=t_eval, method="RK45")
    elif method == "RK4":
        sol = solve_ivp(lambda t, y: riccati_rhs(t, y, a), (t0, tf), [y0], t_eval=t_eval, method="RK45")
    elif method == "RK45":
        sol = solve_ivp(lambda t, y: riccati_rhs(t, y, a), (t0, tf), [y0], t_eval=t_eval, method="RK45", rtol=tol, atol=tol)
    
    st.write(f"Results for parameter a = {a}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], label=f"Solution (method={method})")
    plt.title(f"Solution of Riccati BVP for a={a}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    app()
