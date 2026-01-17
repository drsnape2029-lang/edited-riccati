import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider

# ---------- Original ODE Functions ----------
def f_riccati(x: float, y: float, a: float) -> float:
    """Right-hand side f(x,y) = a*y^2 + 1/a."""
    return a * y**2 + 1.0 / a


def y_exact(x: np.ndarray, a: float) -> np.ndarray:
    """Exact solution y(x) = (1/a) * tan(x)."""
    return (1.0 / a) * np.tan(x)


def yprime_exact(x: np.ndarray, a: float) -> np.ndarray:
    """Exact derivative y'(x) = (1/a) * sec^2(x)."""
    return (1.0 / a) * (1.0 / (np.cos(x) ** 2))


# ---------- Methods for Error Analysis ----------
def compute_errors(y_numerical, y_exact):
    """Compute absolute and relative errors."""
    abs_error = np.abs(y_numerical - y_exact)
    rel_error = abs_error / np.abs(y_exact)
    return abs_error, rel_error


# ---------- Spectral Method (Fourier Series) ----------
def fourier_series(N, a, x):
    """Spectral method using Fourier sine series for the Riccati equation."""
    # Compute Fourier coefficients numerically from the exact solution
    x_fine = np.linspace(0, np.pi, 1000)  # Fine grid for integration
    y_exact_fine = y_exact(x_fine, a)

    y = np.zeros_like(x)
    for n in range(1, N + 1):
        # Compute b_n = (2/π) * ∫_0^π y_exact(x)*sin(n*x) dx using trapezoidal rule
        integrand = y_exact_fine * np.sin(n * x_fine)
        integral = np.trapz(integrand, x_fine)
        b_n = (2.0 / np.pi) * integral
        y += b_n * np.sin(n * x)
    return y


# ---------- Crank-Nicolson Method (Implicit) ----------
def crank_nicolson_step(a, y_prev, h, x_next):
    """Crank-Nicolson method for solving the Riccati equation.
    Solves the implicit equation: y_{n+1} = y_n + (h/2)*(f(x_n,y_n) + f(x_{n+1},y_{n+1}))
    For y' = a*y^2 + 1/a, this gives a quadratic equation."""
    f_prev = f_riccati(x_next, y_prev, a)  # f(x_n, y_n)

    # The implicit equation is:
    # y_next = y_prev + (h/2) * (f_prev + a*y_next^2 + 1/a)
    # Let c = y_prev + (h/2)*(f_prev + 1/a)
    # Then: y_next = c + (h/2)*a*y_next^2
    # So: (h/2)*a*y_next^2 - y_next + c = 0

    c = y_prev + (h/2) * (f_prev + 1.0/a)

    # Quadratic equation: A*y^2 + B*y + C = 0
    A = (h/2) * a
    B = -1
    C = c

    # Solve quadratic equation
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        # Fallback to explicit Euler if no real solution
        return y_prev + h * f_prev

    # Take the root that makes sense for the Riccati equation
    y_next1 = (-B + math.sqrt(discriminant)) / (2*A)
    y_next2 = (-B - math.sqrt(discriminant)) / (2*A)

    # Choose the solution closer to the explicit Euler prediction
    y_euler = y_prev + h * f_prev
    if abs(y_next1 - y_euler) < abs(y_next2 - y_euler):
        return y_next1
    else:
        return y_next2


# ---------- Trigonometric Transformation Method ----------
def riccati_transformed(x, a):
    """Trigonometric transformation method for the Riccati equation.
    Since the exact solution is y(x) = (1/a) * tan(x) = (1/a) * sin(x)/cos(x),
    we use this trigonometric identity."""
    # Avoid division by zero at x = π/2
    safe_x = np.clip(x, 0, np.pi - 1e-10)
    return (1.0 / a) * np.sin(safe_x) / np.cos(safe_x)  # Transformed equation for y(t)


# ---------- Data Calculation Function ----------
def calculate_solutions(a=1, N=10, h=0.01):
    """Calculate solutions using different methods."""
    # Define the domain and parameters
    x_vals = np.linspace(0, np.pi, 100)  # Domain from 0 to pi

    # Calculate the exact solution
    y_exact_vals = y_exact(x_vals, a)

    # Spectral Method (Fourier Series)
    y_fourier = fourier_series(N, a, x_vals)

    # Crank-Nicolson Method
    y_vals_cn = np.zeros_like(x_vals)
    y_vals_cn[0] = 0  # Initial condition y(0) = 0
    for i in range(1, len(x_vals)):
        y_vals_cn[i] = crank_nicolson_step(a, y_vals_cn[i - 1], h, x_vals[i])

    # Trigonometric Transformation Method
    y_transformed = riccati_transformed(x_vals, a)

    # Compute absolute and relative errors
    abs_error_fourier, rel_error_fourier = compute_errors(y_fourier, y_exact_vals)
    abs_error_cn, rel_error_cn = compute_errors(y_vals_cn, y_exact_vals)
    abs_error_transformed, rel_error_transformed = compute_errors(y_transformed, y_exact_vals)

    return (x_vals, y_exact_vals, y_fourier, y_vals_cn, y_transformed,
            abs_error_fourier, rel_error_fourier, abs_error_cn, rel_error_cn,
            abs_error_transformed, rel_error_transformed)


# ---------- Main Plotting Function ----------
def plot_method(a=1, N=10, h=0.01):
    """Plot the solutions and errors."""
    (x_vals, y_exact_vals, y_fourier, y_vals_cn, y_transformed,
     abs_error_fourier, rel_error_fourier, abs_error_cn, rel_error_cn,
     abs_error_transformed, rel_error_transformed) = calculate_solutions(a, N, h)

    # Plot the results
    plt.figure(figsize=(18, 6))

    # Fourier Series Method
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, y_exact_vals, label="Exact Solution", color='k', linestyle='--')
    plt.plot(x_vals, y_fourier, label="Fourier Series Approximation", color='b')
    plt.title("Fourier Series Method")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

    # Crank-Nicolson Method
    plt.subplot(1, 3, 2)
    plt.plot(x_vals, y_exact_vals, label="Exact Solution", color='k', linestyle='--')
    plt.plot(x_vals, y_vals_cn, label="Crank-Nicolson Method", color='g')
    plt.title("Crank-Nicolson Method")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

    # Trigonometric Transformation Method
    plt.subplot(1, 3, 3)
    plt.plot(x_vals, y_exact_vals, label="Exact Solution", color='k', linestyle='--')
    plt.plot(x_vals, y_transformed, label="Transformed Solution", color='r')
    plt.title("Trigonometric Transformation Method")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

    # Display the solutions
    plt.tight_layout()
    plt.show()

    # Plot the errors (absolute and relative errors)
    plt.figure(figsize=(18, 6))

    # Plot Error for Fourier Series (Absolute and Relative)
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, abs_error_fourier, label="Absolute Error", color='b')
    plt.plot(x_vals, rel_error_fourier, label="Relative Error", linestyle='--', color='b')
    plt.title("Error: Fourier Series Method")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()

    # Plot Error for Crank-Nicolson (Absolute and Relative)
    plt.subplot(1, 3, 2)
    plt.plot(x_vals, abs_error_cn, label="Absolute Error", color='g')
    plt.plot(x_vals, rel_error_cn, label="Relative Error", linestyle='--', color='g')
    plt.title("Error: Crank-Nicolson Method")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()

    # Plot Error for Trigonometric Transformation (Absolute and Relative)
    plt.subplot(1, 3, 3)
    plt.plot(x_vals, abs_error_transformed, label="Absolute Error", color='r')
    plt.plot(x_vals, rel_error_transformed, label="Relative Error", linestyle='--', color='r')
    plt.title("Error: Trigonometric Transformation Method")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------- Interactive Plotting with Sliders ----------
def interactive_plot():
    # Calculate initial data
    (x_vals, y_exact_vals, y_fourier, y_vals_cn, y_transformed,
     abs_error_fourier, rel_error_fourier, abs_error_cn, rel_error_cn,
     abs_error_transformed, rel_error_transformed) = calculate_solutions()

    # Create the figure and subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))

    axcolor = 'lightgoldenrodyellow'
    ax_a = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor=axcolor)
    ax_N = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor=axcolor)
    ax_h = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)

    # Create sliders for adjusting parameters
    slider_a = Slider(ax_a, 'a', 0.1, 5.0, valinit=1)
    slider_N = Slider(ax_N, 'N', 1, 50, valinit=10, valstep=1)
    slider_h = Slider(ax_h, 'h', 0.001, 0.1, valinit=0.01)

    # Initialize plots
    line_exact1, = ax1.plot(x_vals, y_exact_vals, 'k--', label="Exact Solution")
    line_fourier1, = ax1.plot(x_vals, y_fourier, 'b-', label="Fourier Series")
    ax1.set_title("Fourier Series Method")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.grid(True)
    ax1.legend()

    line_exact2, = ax2.plot(x_vals, y_exact_vals, 'k--', label="Exact Solution")
    line_cn2, = ax2.plot(x_vals, y_vals_cn, 'g-', label="Crank-Nicolson")
    ax2.set_title("Crank-Nicolson Method")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y(x)")
    ax2.grid(True)
    ax2.legend()

    line_exact3, = ax3.plot(x_vals, y_exact_vals, 'k--', label="Exact Solution")
    line_trans3, = ax3.plot(x_vals, y_transformed, 'r-', label="Transformed")
    ax3.set_title("Trigonometric Transformation Method")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y(x)")
    ax3.grid(True)
    ax3.legend()

    line_abs4, = ax4.plot(x_vals, abs_error_fourier, 'b-', label="Absolute Error")
    line_rel4, = ax4.plot(x_vals, rel_error_fourier, 'b--', label="Relative Error")
    ax4.set_title("Error: Fourier Series Method")
    ax4.set_xlabel("x")
    ax4.set_ylabel("Error")
    ax4.grid(True)
    ax4.legend()

    line_abs5, = ax5.plot(x_vals, abs_error_cn, 'g-', label="Absolute Error")
    line_rel5, = ax5.plot(x_vals, rel_error_cn, 'g--', label="Relative Error")
    ax5.set_title("Error: Crank-Nicolson Method")
    ax5.set_xlabel("x")
    ax5.set_ylabel("Error")
    ax5.grid(True)
    ax5.legend()

    line_abs6, = ax6.plot(x_vals, abs_error_transformed, 'r-', label="Absolute Error")
    line_rel6, = ax6.plot(x_vals, rel_error_transformed, 'r--', label="Relative Error")
    ax6.set_title("Error: Trigonometric Transformation Method")
    ax6.set_xlabel("x")
    ax6.set_ylabel("Error")
    ax6.grid(True)
    ax6.legend()

    # Update function for sliders
    def update(val):
        a = slider_a.val
        N = int(slider_N.val)
        h = slider_h.val

        # Recalculate data
        (x_vals, y_exact_vals, y_fourier, y_vals_cn, y_transformed,
         abs_error_fourier, rel_error_fourier, abs_error_cn, rel_error_cn,
         abs_error_transformed, rel_error_transformed) = calculate_solutions(a, N, h)

        # Update plots
        line_exact1.set_ydata(y_exact_vals)
        line_fourier1.set_ydata(y_fourier)

        line_exact2.set_ydata(y_exact_vals)
        line_cn2.set_ydata(y_vals_cn)

        line_exact3.set_ydata(y_exact_vals)
        line_trans3.set_ydata(y_transformed)

        line_abs4.set_ydata(abs_error_fourier)
        line_rel4.set_ydata(rel_error_fourier)

        line_abs5.set_ydata(abs_error_cn)
        line_rel5.set_ydata(rel_error_cn)

        line_abs6.set_ydata(abs_error_transformed)
        line_rel6.set_ydata(rel_error_transformed)

        # Update y-axis limits dynamically
        for ax in [ax1, ax2, ax3]:
            ax.relim()
            ax.autoscale_view()

        for ax in [ax4, ax5, ax6]:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    # Attach the update function to sliders
    slider_a.on_changed(update)
    slider_N.on_changed(update)
    slider_h.on_changed(update)

    plt.tight_layout()
    plt.show()


# Run interactive plotting
interactive_plot()
