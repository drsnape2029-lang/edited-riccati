"""
Riccati BVP Explorer 
✅ Interactive zoomable Plotly graphs
✅ Fully controllable parameters
✅ LaTeX equation rendering
✅ Light/Dark mode toggle
✅ Input boxes for domain settings
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


from riccati_core import (
    solve_combined, exact_combined, solve_branch, 
    y_exact, yprime_exact, yprime_from_y
)

# Page configuration
st.set_page_config(
    page_title="Riccati BVP Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")

# Fixed dark theme colors
bg_color = '#2b2b2b'
axes_bg = '#2b2b2b'
text_color = '#e8e8e8'
grid_color = '#444444'
accent_color = '#00d4ff'
header_color = '#00d4ff'
card_bg = '#3a3a3a'
border_color = '#555555'
plotly_template = 'plotly_dark'

# Custom CSS for dark theme with fixed header
st.markdown(f"""
    <style>
    .fixed-header {{
        position: sticky;
        top: 0;
        z-index: 100;
        background: linear-gradient(135deg, #1e1e1e 0%, #2b2b2b 100%);
        padding: 1rem 2rem;
        border-bottom: 2px solid {accent_color};
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}
    .main-title {{
        font-size: 2.2rem;
        font-weight: 900;
        color: {accent_color};
        text-align: center;
        margin: 0;
        letter-spacing: 1px;
    }}
    .sub-title {{
        font-size: 1rem;
        color: #888888;
        text-align: center;
        margin-top: 0.3rem;
    }}
    .equation-box {{
        background: linear-gradient(135deg, #1e1e1e 0%, #2b2b2b 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid {border_color};
        margin: 1rem 0;
        text-align: center;
    }}
    .info-box {{
        background-color: {card_bg};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {border_color};
        margin: 1rem 0;
        font-family: 'Consolas', monospace;
        font-size: 0.9rem;
        color: {text_color};
    }}
    .method-section {{
        background-color: {card_bg};
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid {border_color};
        margin: 1.5rem 0;
    }}
    .method-title {{
        color: {accent_color};
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Color palettes - different colors for each graph
COLOR_PALETTES = {
    'euler': ['#2563EB', '#3B82F6', '#60A5FA'],  # Blue shades
    'rk4': ['#F59E0B', '#F97316', '#FB923C'],    # Orange/Amber shades
    'rk45': ['#10B981', '#059669', '#34D399'],   # Green/Emerald shades
    'exact': 'gold',  # Gold for exact solution
    'sing': '#DC2626',  # Red for singularity
}


def plot_graph(x_numerical, y_numerical, x_exact=None, y_exact=None, 
               x_marker=None, y_marker=None, title="Solution Plot",
               method_name="Numerical", show_exact=False, 
               x_domain_max=math.pi, marker_label=None):
    """
    Plot interactive Plotly graph with numerical and optional exact solutions
    
    Parameters:
    -----------
    x_numerical, y_numerical: arrays for numerical solution
    x_exact, y_exact: arrays for exact solution (optional)
    x_marker, y_marker: marker coordinates (optional)
    title: plot title
    method_name: name of the numerical method
    show_exact: whether to show exact solution
    x_domain_max: maximum x value for x-axis
    marker_label: label for marker
    """
    fig = go.Figure()
    
    # Handle NaN gaps in numerical solution (split into segments)
    nan_mask = np.isnan(y_numerical) | np.isnan(x_numerical)
    if np.any(nan_mask):
        # Split into continuous segments
        valid_start = True
        current_x = []
        current_y = []
        for i in range(len(x_numerical)):
            if not nan_mask[i]:
                current_x.append(x_numerical[i])
                current_y.append(y_numerical[i])
            else:
                if current_x:
                    fig.add_trace(go.Scatter(
                        x=current_x, y=current_y,
                        mode='lines',
                        name=method_name,
                        line=dict(width=2.6, color=COLOR_PALETTES.get(method_name.lower(), COLOR_PALETTES['euler'])[0]),
                        showlegend=valid_start
                    ))
                    valid_start = False
                    current_x = []
                    current_y = []
        if current_x:
            fig.add_trace(go.Scatter(
                x=current_x, y=current_y,
                mode='lines',
                name=method_name,
                line=dict(width=2.6, color=COLOR_PALETTES.get(method_name.lower(), COLOR_PALETTES['euler'])[0]),
                showlegend=not valid_start or not show_exact
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x_numerical, y=y_numerical,
            mode='lines',
            name=method_name,
            line=dict(width=2.6, color=COLOR_PALETTES.get(method_name.lower(), COLOR_PALETTES['euler'])[0]),
            showlegend=not show_exact
        ))

    # Add exact solution if provided
    if show_exact and x_exact is not None and y_exact is not None:
        nan_mask_ex = np.isnan(y_exact) | np.isnan(x_exact)
        if np.any(nan_mask_ex):
            valid_start = True
            current_x = []
            current_y = []
            for i in range(len(x_exact)):
                if not nan_mask_ex[i]:
                    current_x.append(x_exact[i])
                    current_y.append(y_exact[i])
                else:
                    if current_x:
                        fig.add_trace(go.Scatter(
                            x=current_x, y=current_y,
                            mode='lines',
                            name='Exact',
                            line=dict(width=2.2, dash='dot', color=COLOR_PALETTES['exact']),
                            showlegend=valid_start
                        ))
                        valid_start = False
                        current_x = []
                        current_y = []
            if current_x:
                fig.add_trace(go.Scatter(
                    x=current_x, y=current_y,
                    mode='lines',
                    name='Exact',
                    line=dict(width=2.2, dash='dot', color=COLOR_PALETTES['exact']),
                    showlegend=not valid_start
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x_exact, y=y_exact,
                mode='lines',
                name='Exact',
                line=dict(width=2.2, dash='dot', color=COLOR_PALETTES['exact'])
            ))


    # Add singularity marker at x = π/2
    fig.add_vline(
        x=math.pi/2,
        line_dash="dash",
        line_color=COLOR_PALETTES["sing"],
        line_width=2,
        annotation_text="singularity π/2",
        annotation_position="top"
    )

    # Add x marker if provided
    if x_marker is not None and y_marker is not None:
        fig.add_trace(go.Scatter(
            x=[x_marker], y=[y_marker],
            mode='markers',
            name=marker_label if marker_label else f'x={x_marker:.3f}',
            marker=dict(size=12, color='red', symbol='triangle-down', line=dict(width=2, color='darkred'))
        ))

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=accent_color)),
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[0, x_domain_max], gridcolor=grid_color, showgrid=True, gridwidth=1),
        yaxis=dict(range=[-15, 15], gridcolor=grid_color, showgrid=True, gridwidth=1),  # Fixed scale
        template=plotly_template,
        hovermode='x unified',
        legend=dict(bgcolor=card_bg, bordercolor=border_color, borderwidth=1),
        height=600,
        plot_bgcolor=axes_bg,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, size=12)
    )

    return fig


def relative_error_stats(x, y, a, eps=1e-12):
    """Compute relative error statistics"""
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")
    xf = x[m]
    yf = y[m]
    ye = y_exact(xf, a)
    r = np.abs(yf - ye) / (np.abs(ye) + eps)
    return float(np.max(r)), float(np.mean(r)), float(np.sqrt(np.mean(r**2)))




def main():
    """Main Streamlit application"""
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        # Domain controls - INPUT BOXES instead of sliders
        st.subheader("Domain Settings")
        left_frac = st.number_input("Left Fraction (default 0.99)", 
                                    min_value=0.5, max_value=0.999, value=0.99, step=0.001,
                                    format="%.3f",
                                    help="Fraction of π/2 for left domain boundary")
        right_frac = st.number_input("Right Fraction (default 1.01)",
                                     min_value=1.001, max_value=1.5, value=1.01, step=0.001,
                                     format="%.3f",
                                     help="Fraction of π/2 for right domain boundary")
        x_domain_max = math.pi  # Fixed to π
        
        st.divider()
        
        # Numerical method controls
        st.subheader("Numerical Settings")
        h_size = st.number_input("Step Size (h)", 
                                min_value=0.001, max_value=0.1, value=0.05, step=0.001,
                                format="%.3f",
                                help="Step size for Euler and RK4 methods")
        tol = st.number_input("Tolerance (RK45)", 
                             min_value=1e-10, max_value=1e-3, value=1e-6, step=1e-7,
                             format="%.0e",
                             help="Tolerance for RK45 adaptive method")
    
    # Fixed header at the top (Facebook-style)
    st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">⚡ RICCATI BVP EXPLORER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">By: Bader Muneer          Email:Bdr0222060@ju.edu.jo          Instructor: Prof. Zaer Abu Hammour          Email:Zaer@ju.edu.jo        (2026)</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate domain boundaries
    x_left_end = left_frac * math.pi / 2
    x_right_start = right_frac * math.pi / 2
    
    # Create tabs - 4 tabs now
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Mathematical Background", "🧠 Solution", "📊 Comparing Methods", "🎚️ Slider (a)"])
    
    # ================= Tab 1: Mathematical Background =================
    with tab1:
        st.markdown('<div class="method-section">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">Problem Formulation</div>', unsafe_allow_html=True)
        
        st.markdown("### Riccati Boundary-Value Problem")
        st.latex(r"""
        y'(x) = a \cdot y(x)^2 + \frac{1}{a}, \quad x \in [0, \pi], \quad a \neq 0
        """)
        
        st.markdown("### Exact Solution")
        st.latex(r"""
        y(x) = \frac{1}{a} \tan(x)
        """)
        
        st.markdown("### Exact Derivative")
        st.latex(r"""
        y'(x) = \frac{1}{a} \sec^2(x) = \frac{1}{a \cos^2(x)}
        """)
        
        st.markdown("### Domain and Singularity")
        st.write(r"The exact solution has a **singularity at** $x = \frac{\pi}{2}$ where $\tan(x) \to \pm\infty$.")
        left_str = f"{left_frac:.2f}"
        right_str = f"{right_frac:.2f}"
        st.write(f"To avoid the singularity, we solve on the domain:")
        st.latex(f"\\text{{Domain: }} [0, {left_str} \\cdot \\frac{{\\pi}}{{2}}] \\cup [{right_str} \\cdot \\frac{{\\pi}}{{2}}, \\pi]")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="method-section">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">Numerical Methods</div>', unsafe_allow_html=True)
        
        # Euler Method
        st.markdown("#### 1. Euler Method (Explicit Euler)")
        st.latex(r"""
        y_{n+1} = y_n + h \cdot f(x_n, y_n)
        """)
        st.latex(r"""
        \text{where } f(x, y) = a \cdot y^2 + \frac{1}{a}
        """)
        st.write("**Error:** $O(h)$ (first-order accurate)")
        st.write("**Advantages:** Simple, computationally efficient")
        st.write("**Disadvantages:** Low accuracy, requires small step sizes")
        
        st.divider()
        
        # RK4 Method
        st.markdown("#### 2. Runge-Kutta 4th Order (RK4)")
        st.latex(r"""
        \begin{aligned}
        k_1 &= h \cdot f(x_n, y_n) \\
        k_2 &= h \cdot f(x_n + \frac{h}{2}, y_n + \frac{k_1}{2}) \\
        k_3 &= h \cdot f(x_n + \frac{h}{2}, y_n + \frac{k_2}{2}) \\
        k_4 &= h \cdot f(x_n + h, y_n + k_3) \\
        y_{n+1} &= y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
        \end{aligned}
        """)
        st.write("**Error:** $O(h^4)$ (fourth-order accurate)")
        st.write("**Advantages:** High accuracy, stable for most problems")
        st.write("**Disadvantages:** More function evaluations per step")
        
        st.divider()
        
        # RK45 Method
        st.markdown("#### 3. Runge-Kutta-Fehlberg (RK45) - Adaptive")
        st.write("Uses a 5th-order method to estimate error and adjust step size:")
        st.latex(r"""
        \text{Step size: } h_{new} = 0.9 \cdot h \cdot \left(\frac{\text{tol}}{|\text{error}|}\right)^{0.25}
        """)
        st.write("**Error:** Controlled by tolerance parameter")
        st.write("**Advantages:** Automatic step size adjustment, efficient for varying solution behavior")
        st.write("**Disadvantages:** More complex implementation, variable step sizes")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="method-section">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">Solution Strategy</div>', unsafe_allow_html=True)
        
        st.markdown("### Left Branch Solution")
        st.write(f"Solve from $x = 0$ to $x = {left_str} \\cdot \\frac{{\\pi}}{{2}}$:")
        st.latex(r"""
        y(0) = 0, \quad \text{(initial condition)}
        """)
        st.write("Integrate forward using chosen method.")
        
        st.markdown("### Right Branch Solution")
        st.write(f"Solve from $x = \\pi$ to $x = {right_str} \\cdot \\frac{{\\pi}}{{2}}$:")
        st.latex(r"""
        y(\pi) = 0, \quad \text{(boundary condition)}
        """)
        st.write("Integrate backward (reverse direction) using chosen method.")
        
        st.markdown("### Combined Solution")
        st.write("The two branches are combined with a NaN gap at the singularity for clean visualization:")
        combined_eq = rf"""
        y_{{\text{{combined}}}}(x) = \begin{{cases}}
        y_{{\text{{left}}}}(x) & \text{{for }} x \in [0, {left_str}\cdot\frac{{\pi}}{{2}}] \\
        \text{{NaN}} & \text{{for }} x \in [{left_str}\cdot\frac{{\pi}}{{2}}, {right_str}\cdot\frac{{\pi}}{{2}}] \\
        y_{{\text{{right}}}}(x) & \text{{for }} x \in [{right_str}\cdot\frac{{\pi}}{{2}}, \pi]
        \end{{cases}}
        """
        st.latex(combined_eq)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="method-section">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">Error Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("### Absolute Error")
        st.latex(r"""
        E_{\text{abs}}(x) = |y_{\text{numerical}}(x) - y_{\text{exact}}(x)|
        """)
        
        st.markdown("### Relative Error")
        st.latex(r"""
        E_{\text{rel}}(x) = \frac{|y_{\text{numerical}}(x) - y_{\text{exact}}(x)|}{|y_{\text{exact}}(x)| + \epsilon}
        """)
        st.write("where $\\epsilon$ is a small number to avoid division by zero.")
        
        st.markdown("### Global Error Statistics")
        st.latex(r"""
        \begin{aligned}
        E_{\max} &= \max_x E_{\text{rel}}(x) \\
        E_{\text{mean}} &= \frac{1}{N} \sum_{i=1}^{N} E_{\text{rel}}(x_i) \\
        E_{\text{RMS}} &= \sqrt{\frac{1}{N} \sum_{i=1}^{N} E_{\text{rel}}^2(x_i)}
        \end{aligned}
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ================= Tab 2: Solution (formerly Tab 1) =================
    
    # ================= Tab 2: Solution =================
    with tab2:
        st.subheader("Single Method Solution")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            a1 = st.number_input("Parameter a", value=2.0, min_value=0.1, max_value=10.0, 
                                step=0.1, key="a1", format="%.2f")
        
        with col2:
            method1 = st.selectbox("Method", ["Euler", "RK4", "RK45"], key="method1")
        
        with col3:
            show_exact1 = st.checkbox("Show Exact", value=True, key="exact1")
        
        with col4:
            use_x1 = st.checkbox("Use x marker", value=True, key="usex1")
        
        x1_val = None
        if use_x1:
            x1_val = st.number_input("x value", value=1.0, min_value=0.0, 
                                    max_value=float(x_domain_max), step=0.1, key="x1")
        
        autofocus1 = st.checkbox("Focus near x", value=True, key="focus1")
        
        # Auto-plot on parameter change
        if abs(a1) >= 1e-12:
            method_map = {"Euler": "euler", "RK4": "rk4", "RK45": "rk45"}
            method_key = method_map[method1]
            
            # Solve numerical solution
            x_num, y_num = solve_combined(
                a1, method_key, h=h_size, tol=tol,
                left_frac=left_frac, right_frac=right_frac, 
                x_domain_max=x_domain_max, gap_nan=True
            )
            
            # Solve exact solution if needed
            x_ex = None
            y_ex = None
            if show_exact1:
                x_ex, y_ex = exact_combined(
                    a1, left_frac=left_frac, right_frac=right_frac, 
                    x_domain_max=x_domain_max, n=1400
                )
            
            # Handle x marker
            x_marker = None
            y_marker = None
            marker_label = None
            if use_x1 and x1_val is not None:
                if 0 <= x1_val <= x_domain_max and not (x_left_end < x1_val < x_right_start):
                    y_marker = float(np.interp(x1_val, x_num[~np.isnan(x_num)], y_num[~np.isnan(y_num)]))
                    x_marker = x1_val
                    marker_label = f'x={x1_val:.3f}'
            
            # Create plot
            fig = plot_graph(
                x_num, y_num, x_ex, y_ex,
                x_marker, y_marker,
                title=f"{method1} Solution (a={a1})",
                method_name=method1,
                show_exact=show_exact1,
                x_domain_max=x_domain_max,
                marker_label=marker_label
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Status info box
            info_lines = [
                f"**Parameter a:** {a1:.4f}",
                f"**Method:** {method1}",
                f"**Domain:** [0, {left_frac:.3f}·π/2] ∪ [{right_frac:.3f}·π/2, {x_domain_max:.3f}]",
                f"**Step Size (h):** {h_size:.3f}",
                f"**Tolerance (RK45):** {tol:.0e}"
            ]
            
            if use_x1 and x1_val is not None:
                if x_left_end < x1_val < x_right_start:
                    info_lines.append("⚠️ **x is inside forbidden gap near π/2 → skipped**")
                elif x1_val < 0 or x1_val > x_domain_max:
                    info_lines.append(f"⚠️ **x outside [0, {x_domain_max:.3f}] → skipped**")
                else:
                    branch = "left" if x1_val <= x_left_end else "right"
                    xb, yb = solve_branch(
                        a1, method_key, branch=branch, h=h_size, tol=tol,
                        left_frac=left_frac, right_frac=right_frac, 
                        x_domain_max=x_domain_max
                    )
                    y_num_val = float(np.interp(x1_val, xb, yb))
                    info_lines.append(f"**At x = {x1_val:.6f}:** y_num = {y_num_val:.6e}")
                    if show_exact1:
                        y_exact_val = float(y_exact(np.array([x1_val]), a1)[0])
                        err_abs = abs(y_num_val - y_exact_val)
                        err_rel = err_abs / (abs(y_exact_val) + 1e-12)
                        info_lines.append(f"**Exact:** y = {y_exact_val:.6e} | **|Error|** = {err_abs:.3e} | **Rel Error** = {err_rel:.3e}")
            
            st.markdown(f'<div class="info-box">{"<br>".join(info_lines)}</div>', 
                       unsafe_allow_html=True)
        
    
    # ================= Tab 3: Comparing Methods =================
    with tab3:
        st.subheader("Method Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a2 = st.number_input("Parameter a", value=2.0, min_value=0.1, max_value=10.0, 
                                step=0.1, key="a2", format="%.2f")
        
        with col2:
            x2_val = st.text_input("x (optional)", value="", key="x2", 
                                   help="Optional x value for marker")
        
        with col3:
            show_exact2 = st.checkbox("Show Exact", value=True, key="exact2")
        
        # Auto-plot on parameter change
        if abs(a2) >= 1e-12:
            x_val = None
            if x2_val.strip():
                try:
                    x_val = float(x2_val)
                    if x_val < 0 or x_val > x_domain_max:
                        st.error(f"x must be in [0, {x_domain_max:.3f}]")
                        x_val = None
                except:
                    x_val = None
            
            # Solution plots (side by side) - Interactive Plotly
            fig_solutions = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Euler Solution", "RK4 Solution", "RK45 Solution"),
                shared_yaxes=True
            )
            
            methods_data = [
                ("euler", "Euler", 1),
                ("rk4", "RK4", 2),
                ("rk45", "RK45", 3)
            ]
            
            results = {}
            
            for method_key, method_name, col_idx in methods_data:
                x, y = solve_combined(a2, method_key, h=h_size, tol=tol,
                                     left_frac=left_frac, right_frac=right_frac,
                                     x_domain_max=x_domain_max, gap_nan=True)
                
                # Handle NaN gaps
                nan_mask = np.isnan(y) | np.isnan(x)
                if np.any(nan_mask):
                    valid_start = True
                    current_x = []
                    current_y = []
                    for i in range(len(x)):
                        if not nan_mask[i]:
                            current_x.append(x[i])
                            current_y.append(y[i])
                        else:
                            if current_x:
                                fig_solutions.add_trace(
                                    go.Scatter(x=current_x, y=current_y, mode='lines',
                                              name=method_name, line=dict(width=2.6, 
                                              color=COLOR_PALETTES[method_key][0]),
                                              showlegend=(col_idx == 1 and not show_exact2)),
                                    row=1, col=col_idx
                                )
                                valid_start = False
                                current_x = []
                                current_y = []
                    if current_x:
                        fig_solutions.add_trace(
                            go.Scatter(x=current_x, y=current_y, mode='lines',
                                      name=method_name, line=dict(width=2.6,
                                      color=COLOR_PALETTES[method_key][0]),
                                      showlegend=(col_idx == 1 and not show_exact2)),
                            row=1, col=col_idx
                        )
                else:
                    fig_solutions.add_trace(
                        go.Scatter(x=x, y=y, mode='lines',
                                  name=method_name, line=dict(width=2.6,
                                  color=COLOR_PALETTES[method_key][0]),
                                  showlegend=(col_idx == 1 and not show_exact2)),
                        row=1, col=col_idx
                    )
                
                if show_exact2:
                    x_ex, y_ex = exact_combined(a2, left_frac=left_frac, right_frac=right_frac,
                                               x_domain_max=x_domain_max, n=1400)
                    nan_mask_ex = np.isnan(y_ex) | np.isnan(x_ex)
                    if np.any(nan_mask_ex):
                        current_x = []
                        current_y = []
                        for i in range(len(x_ex)):
                            if not nan_mask_ex[i]:
                                current_x.append(x_ex[i])
                                current_y.append(y_ex[i])
                            else:
                                if current_x:
                                    fig_solutions.add_trace(
                                        go.Scatter(x=current_x, y=current_y, mode='lines',
                                                  name='Exact', line=dict(width=2.2, dash='dot',
                                                  color=COLOR_PALETTES['exact']),
                                                  showlegend=(col_idx == 1)),
                                        row=1, col=col_idx
                                    )
                                    current_x = []
                                    current_y = []
                        if current_x:
                            fig_solutions.add_trace(
                                go.Scatter(x=current_x, y=current_y, mode='lines',
                                          name='Exact', line=dict(width=2.2, dash='dot',
                                          color=COLOR_PALETTES['exact']),
                                          showlegend=(col_idx == 1)),
                                row=1, col=col_idx
                            )
                    else:
                        fig_solutions.add_trace(
                            go.Scatter(x=x_ex, y=y_ex, mode='lines',
                                      name='Exact', line=dict(width=2.2, dash='dot',
                                      color=COLOR_PALETTES['exact']),
                                      showlegend=(col_idx == 1)),
                            row=1, col=col_idx
                        )
                
                # Add singularity marker
                fig_solutions.add_vline(x=math.pi/2, line_dash="dash", line_color=COLOR_PALETTES["sing"],
                                       line_width=2, row=1, col=col_idx)
                
                # Add x marker if provided
                if x_val is not None and 0 <= x_val <= x_domain_max and not (x_left_end < x_val < x_right_start):
                    y_at_x = float(np.interp(x_val, x[~np.isnan(x)], y[~np.isnan(y)]))
                    fig_solutions.add_trace(
                        go.Scatter(x=[x_val], y=[y_at_x], mode='markers',
                                  name=f'x={x_val:.3f}' if col_idx == 1 else None,
                                  marker=dict(size=12, color='red', symbol='triangle-down'),
                                  showlegend=(col_idx == 1)),
                        row=1, col=col_idx
                    )
                
                results[method_key] = (x, y)
            
            fig_solutions.update_xaxes(title_text="x", range=[0, x_domain_max], gridcolor=grid_color, 
                                      showgrid=True, row=1, col=1)
            fig_solutions.update_xaxes(title_text="x", range=[0, x_domain_max], gridcolor=grid_color, 
                                      showgrid=True, row=1, col=2)
            fig_solutions.update_xaxes(title_text="x", range=[0, x_domain_max], gridcolor=grid_color, 
                                      showgrid=True, row=1, col=3)
            fig_solutions.update_yaxes(title_text="y", range=[-15, 15], gridcolor=grid_color, showgrid=True)  # Fixed scale
            fig_solutions.update_layout(
                height=600,
                template=plotly_template,
                plot_bgcolor=axes_bg,
                paper_bgcolor=bg_color,
                font=dict(color=text_color, size=12),
                legend=dict(bgcolor=card_bg, bordercolor=border_color, borderwidth=1)
            )
            
            st.plotly_chart(fig_solutions, use_container_width=True)
            
            # Combined error plot - Interactive Plotly
            fig_err = go.Figure()
            
            maxr_e = 0.0
            maxr_4 = 0.0
            maxr_5 = 0.0
            
            for method_key, method_name in [("euler", "Euler"), ("rk4", "RK4"), ("rk45", "RK45")]:
                x, y = results[method_key]
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    xf = x[m]
                    yf = y[m]
                    ye = y_exact(xf, a2)
                    r = np.abs(yf - ye) / (np.abs(ye) + 1e-12)
                    # Use first color from each method's palette for consistency
                    color = COLOR_PALETTES[method_key][0]
                    fig_err.add_trace(go.Scatter(
                        x=xf, y=r,
                        mode='lines',
                        name=method_name,
                        line=dict(width=2.5, color=color)
                    ))
                    max_r = float(np.max(r))
                    if method_key == "euler":
                        maxr_e = max_r
                    elif method_key == "rk4":
                        maxr_4 = max_r
                    else:
                        maxr_5 = max_r
            
            fig_err.add_vline(x=math.pi/2, line_dash="dash", line_color=COLOR_PALETTES["sing"],
                             line_width=2, annotation_text="singularity π/2")
            
            fig_err.update_layout(
                title=dict(text="Relative Errors Comparison", font=dict(size=16, color=accent_color)),
                xaxis_title="x",
                yaxis_title="Relative Error |yₙᵤₘ - yₑₓₐcₜ| / |yₑₓₐcₜ|",
                yaxis_type="log",
                xaxis=dict(range=[0, x_domain_max], gridcolor=grid_color, showgrid=True),
                yaxis=dict(
                    autorange=True,  # Let plotly auto-scale for log scale
                    gridcolor=grid_color,
                    showgrid=True,
                    tickformat=".0e"  # Scientific notation for log scale
                ),
                template=plotly_template,
                height=600,
                plot_bgcolor=axes_bg,
                paper_bgcolor=bg_color,
                font=dict(color=text_color, size=12),
                legend=dict(bgcolor=card_bg, bordercolor=border_color, borderwidth=1)
            )
            
            st.plotly_chart(fig_err, use_container_width=True)
            
            # Info box
            info_lines = [
                f"**Parameter a:** {a2:.4f}",
                f"**Domain:** [0, {left_frac:.3f}·π/2] ∪ [{right_frac:.3f}·π/2, {x_domain_max:.3f}]",
                f"**Step Size (h):** {h_size:.3f} | **Tolerance:** {tol:.0e}",
                ""
            ]
            
            if x_val is not None:
                info_lines.extend([
                    f"**At x = {x_val:.6f}:**",
                    "| Method    | y_num      | y_exact    | Error      | Rel Error |",
                    "|-----------|------------|------------|------------|-----------|"
                ])
                
                y_exact_val = float(y_exact(np.array([x_val]), a2)[0])
                
                for method_key, method_name in [("euler", "Euler"), ("rk4", "RK4"), ("rk45", "RK45")]:
                    try:
                        x, y = solve_combined(a2, method_key, h=h_size, tol=tol,
                                             left_frac=left_frac, right_frac=right_frac,
                                             x_domain_max=x_domain_max, gap_nan=False)
                        y_num_val = float(np.interp(x_val, x, y))
                        error = abs(y_num_val - y_exact_val)
                        rel_error = error / (abs(y_exact_val) + 1e-12)
                        info_lines.append(
                            f"| {method_name:<9} | {y_num_val:>10.6e} | {y_exact_val:>10.6e} | {error:>10.3e} | {rel_error:>8.3e} |"
                        )
                    except Exception as e:
                        info_lines.append(f"| {method_name:<9} | Error: {str(e)[:30]} |")
                
                info_lines.append("")
            
            info_lines.extend([
                "**Global Relative Error Statistics:**",
                f"| Method    | Max Error    |",
                "|-----------|--------------|",
                f"| Euler     | {maxr_e:>12.3e} |",
                f"| RK4       | {maxr_4:>12.3e} |",
                f"| RK45      | {maxr_5:>12.3e} |"
            ])
            
            st.markdown(f'<div class="info-box">{"<br>".join(info_lines)}</div>', 
                       unsafe_allow_html=True)
        
    
    # ================= Tab 4: Slider (a) =================
    with tab4:
        st.subheader("Interactive Parameter Slider")
        
        col1, col2, col3 = st.columns([2, 1.5, 2])
        
        with col1:
            multi_a_mode = st.checkbox("Multi-a Mode", value=False, 
                                      help="Plot multiple a values on same axes")
        
        with col2:
            if not multi_a_mode:
                a3 = st.slider("Parameter a", min_value=1, max_value=10, value=2, 
                              step=1, key="a3_slider")
                a3_val = float(a3)
                st.write(f"**Current a: {a3_val:.1f}**")
            else:
                a_multi_input = st.text_input("a values (comma-separated)", 
                                             value="-5,-2,-1,1,2,5", key="a_multi")
                try:
                    a_values = [float(s.strip()) for s in a_multi_input.split(',') if s.strip()]
                    a3_val = a_values[0] if a_values else 2.0
                except:
                    st.error("Invalid a values format. Use comma-separated numbers.")
                    a_values = [2.0]
                    a3_val = 2.0
        
        with col3:
            method3 = st.selectbox("Method", ["Euler", "RK4", "RK45", "All"], key="method3")
            show_exact3 = st.checkbox("Show Exact", value=True, key="exact3")
        
        # Auto-update on parameter change - Interactive Plotly
        fig3 = go.Figure()
        
        if multi_a_mode:
            colors = ['#2563EB', '#10B981', '#F59E0B', '#DC2626', '#8B5CF6', '#EC4899', '#06B6D4']
            
            if method3 == "Euler":
                methods = [("euler", "Euler")]
            elif method3 == "RK4":
                methods = [("rk4", "RK4")]
            elif method3 == "RK45":
                methods = [("rk45", "RK45")]
            else:
                methods = [("euler", "Euler"), ("rk4", "RK4"), ("rk45", "RK45")]
            
            for i, a_val in enumerate(a_values):
                if abs(a_val) < 1e-12:
                    continue
                color = colors[i % len(colors)]
                
                method_color_idx = 0
                for key, name in methods:
                    label = f"{name} (a={a_val})" if len(methods) > 1 else f"a={a_val}"
                    x, y = solve_combined(a_val, key, h=h_size, tol=tol,
                                         left_frac=left_frac, right_frac=right_frac,
                                         x_domain_max=x_domain_max, gap_nan=True)

                    plot_color = COLOR_PALETTES[key][method_color_idx % len(COLOR_PALETTES[key])] if len(methods) > 1 else color

                    # Handle NaN gaps - only show legend for the first segment
                    nan_mask = np.isnan(y) | np.isnan(x)
                    if np.any(nan_mask):
                        current_x = []
                        current_y = []
                        first_segment = True
                        for j in range(len(x)):
                            if not nan_mask[j]:
                                current_x.append(x[j])
                                current_y.append(y[j])
                            else:
                                if current_x:
                                    fig3.add_trace(go.Scatter(
                                        x=current_x, y=current_y, mode='lines',
                                        name=label if first_segment else None,
                                        line=dict(width=2.6, color=plot_color),
                                        showlegend=first_segment
                                    ))
                                    first_segment = False
                                    current_x = []
                                    current_y = []
                        if current_x:
                            fig3.add_trace(go.Scatter(
                                x=current_x, y=current_y, mode='lines',
                                name=label if first_segment else None,
                                line=dict(width=2.6, color=plot_color),
                                showlegend=first_segment
                            ))
                    else:
                        fig3.add_trace(go.Scatter(
                            x=x, y=y, mode='lines',
                            name=label, line=dict(width=2.6, color=plot_color)
                        ))
                    method_color_idx += 1
                
                if show_exact3:
                    x_ex, y_ex = exact_combined(a_val, left_frac=left_frac, right_frac=right_frac,
                                               x_domain_max=x_domain_max, n=1400)
                    exact_label = f"Exact (a={a_val})" if len(a_values) > 1 else "Exact"
                    nan_mask_ex = np.isnan(y_ex) | np.isnan(x_ex)
                    if np.any(nan_mask_ex):
                        current_x = []
                        current_y = []
                        first_segment = True
                        for j in range(len(x_ex)):
                            if not nan_mask_ex[j]:
                                current_x.append(x_ex[j])
                                current_y.append(y_ex[j])
                            else:
                                if current_x:
                                    fig3.add_trace(go.Scatter(
                                        x=current_x, y=current_y, mode='lines',
                                        name=exact_label if first_segment else None,
                                        line=dict(width=2.2, dash='dash', color=color, opacity=0.7),
                                        showlegend=first_segment
                                    ))
                                    first_segment = False
                                    current_x = []
                                    current_y = []
                        if current_x:
                            fig3.add_trace(go.Scatter(
                                x=current_x, y=current_y, mode='lines',
                                name=exact_label if first_segment else None,
                                line=dict(width=2.2, dash='dash', color=color, opacity=0.7),
                                showlegend=first_segment
                            ))
                    else:
                        fig3.add_trace(go.Scatter(
                            x=x_ex, y=y_ex, mode='lines',
                            name=exact_label,
                            line=dict(width=2.2, dash='dash', color=color, opacity=0.7)
                        ))
        else:
            if method3 == "Euler":
                methods = [("euler", "Euler")]
            elif method3 == "RK4":
                methods = [("rk4", "RK4")]
            elif method3 == "RK45":
                methods = [("rk45", "RK45")]
            else:
                methods = [("euler", "Euler"), ("rk4", "RK4"), ("rk45", "RK45")]
            
            color_idx = 0
            for key, name in methods:
                x, y = solve_combined(a3_val, key, h=h_size, tol=tol,
                                     left_frac=left_frac, right_frac=right_frac,
                                     x_domain_max=x_domain_max, gap_nan=True)

                # Handle NaN gaps - only show legend for the first segment
                nan_mask = np.isnan(y) | np.isnan(x)
                if np.any(nan_mask):
                    current_x = []
                    current_y = []
                    first_segment = True
                    for j in range(len(x)):
                        if not nan_mask[j]:
                            current_x.append(x[j])
                            current_y.append(y[j])
                        else:
                            if current_x:
                                fig3.add_trace(go.Scatter(
                                    x=current_x, y=current_y, mode='lines',
                                    name=name if first_segment else None,
                                    line=dict(width=2.6,
                                    color=COLOR_PALETTES[key][color_idx % len(COLOR_PALETTES[key])]),
                                    showlegend=first_segment
                                ))
                                first_segment = False
                                current_x = []
                                current_y = []
                    if current_x:
                        fig3.add_trace(go.Scatter(
                            x=current_x, y=current_y, mode='lines',
                            name=name if first_segment else None,
                            line=dict(width=2.6,
                            color=COLOR_PALETTES[key][color_idx % len(COLOR_PALETTES[key])]),
                            showlegend=first_segment
                        ))
                else:
                    fig3.add_trace(go.Scatter(
                        x=x, y=y, mode='lines',
                        name=name, line=dict(width=2.6,
                        color=COLOR_PALETTES[key][color_idx % len(COLOR_PALETTES[key])])
                    ))
                color_idx += 1
            
            if show_exact3:
                x_ex, y_ex = exact_combined(a3_val, left_frac=left_frac, right_frac=right_frac,
                                           x_domain_max=x_domain_max, n=1400)
                nan_mask_ex = np.isnan(y_ex) | np.isnan(x_ex)
                if np.any(nan_mask_ex):
                    current_x = []
                    current_y = []
                    first_segment = True
                    for j in range(len(x_ex)):
                        if not nan_mask_ex[j]:
                            current_x.append(x_ex[j])
                            current_y.append(y_ex[j])
                        else:
                            if current_x:
                                fig3.add_trace(go.Scatter(
                                    x=current_x, y=current_y, mode='lines',
                                    name='Exact' if first_segment else None,
                                    line=dict(width=2.2, dash='dot',
                                    color=COLOR_PALETTES['exact']),
                                    showlegend=first_segment
                                ))
                                first_segment = False
                                current_x = []
                                current_y = []
                    if current_x:
                        fig3.add_trace(go.Scatter(
                            x=current_x, y=current_y, mode='lines',
                            name='Exact' if first_segment else None,
                            line=dict(width=2.2, dash='dot',
                            color=COLOR_PALETTES['exact']),
                            showlegend=first_segment
                        ))
                else:
                    fig3.add_trace(go.Scatter(
                        x=x_ex, y=y_ex, mode='lines',
                        name='Exact', line=dict(width=2.2, dash='dot',
                        color=COLOR_PALETTES['exact'])
                    ))
        
        # Add singularity marker
        fig3.add_vline(x=math.pi/2, line_dash="dash", line_color=COLOR_PALETTES["sing"],
                      line_width=2, annotation_text="singularity π/2", annotation_position="top")
        
        fig3.update_layout(
            title=dict(text=f"{method3} — Fixed Scale", font=dict(size=16, color=accent_color)),
            xaxis_title="x",
            yaxis_title="y",
            xaxis=dict(range=[0, x_domain_max], gridcolor=grid_color, showgrid=True),
            yaxis=dict(range=[-15, 15], gridcolor=grid_color, showgrid=True),  # Fixed scale
            template=plotly_template,
            height=600,
            plot_bgcolor=axes_bg,
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            legend=dict(bgcolor=card_bg, bordercolor=border_color, borderwidth=1)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Info box for Tab 4
        if multi_a_mode:
            info_lines = [
                "**Relative Error Statistics for Different a Values:**",
                "| a Value   | Max Error    |",
                "|-----------|--------------|"
            ]
            for a_val in a_values:
                if abs(a_val) < 1e-12:
                    continue
                try:
                    x, y = solve_combined(a_val, "rk45", h=h_size, tol=tol,
                                         left_frac=left_frac, right_frac=right_frac,
                                         x_domain_max=x_domain_max, gap_nan=False)
                    maxr, _, _ = relative_error_stats(x, y, a_val, eps=1e-12)
                    info_lines.append(f"| {a_val:>8.1f} | {maxr:>12.3e} |")
                except:
                    info_lines.append(f"| {a_val:>8.1f} | Error         |")
            st.markdown(f'<div class="info-box">{"<br>".join(info_lines)}</div>', 
                       unsafe_allow_html=True)
        


if __name__ == "__main__":
    main()
