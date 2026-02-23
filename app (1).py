import streamlit as st
import numpy as np
import plotly.graph_objects as go
# Your Bloch sphere implementation here

st.set_page_config(page_title="Interactive 3D Bloch Sphere", layout="wide")
st.title("Interactive 3D Bloch Sphere")

# ── Gate definitions ──────────────────────────────────────────────────────────
pauli_x_gate  = np.array([[0, 1],  [1, 0]],          dtype=complex)
pauli_y_gate  = np.array([[0, -1j],[1j, 0]],          dtype=complex)
pauli_z_gate  = np.array([[1, 0],  [0, -1]],          dtype=complex)
hadamard_gate = np.array([[1, 1],  [1, -1]],          dtype=complex) / np.sqrt(2)
S_gate        = np.array([[1, 0],  [0, 1j]],          dtype=complex)
Sdag_gate     = np.array([[1, 0],  [0, -1j]],         dtype=complex)
T_gate        = np.array([[1, 0],  [0, np.exp(1j*np.pi/4)]],  dtype=complex)
Tdag_gate     = np.array([[1, 0],  [0, np.exp(-1j*np.pi/4)]], dtype=complex)

buttons = {
    "Pauli X": pauli_x_gate,
    "Pauli Y": pauli_y_gate,
    "Pauli Z": pauli_z_gate,
    "S": S_gate,
    "S†": Sdag_gate,
    "T": T_gate,
    "T†": Tdag_gate,
    "H": hadamard_gate,
}

circuit_inputs = {
    "X": pauli_x_gate,
    "Y": pauli_y_gate,
    "Z": pauli_z_gate,
    "S": S_gate,
    "Sdagger": Sdag_gate,
    "T": T_gate,
    "Tdagger": Tdag_gate,
    "H": hadamard_gate,
}

preset_states = {
    "|0⟩ — spin up":    (0.0,      0.0),
    "|1⟩ — spin down":  (np.pi,    0.0),
    "|+⟩ — Hadamard +": (np.pi/2,  0.0),
    "|-⟩ — Hadamard -": (np.pi/2,  np.pi),
}

# ── Session state initialisation ──────────────────────────────────────────────
if "theta" not in st.session_state:
    st.session_state.theta = np.pi / 4
if "phi" not in st.session_state:
    st.session_state.phi = np.pi / 4
if "slerp_history" not in st.session_state:
    st.session_state.slerp_history = []
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "radians"

# ── Math helpers ──────────────────────────────────────────────────────────────
def angles_to_state(theta, phi):
    return np.array([
        [np.cos(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2)]], dtype=complex)

def state_to_angles(state):
    state = state / np.linalg.norm(state)
    alpha, beta = state[0, 0], state[1, 0]
    theta = 2 * np.arccos(np.clip(np.abs(alpha), 0, 1))
    phi = np.angle(beta) - np.angle(alpha)
    phi = (phi + 2*np.pi) % (2*np.pi)
    return float(theta.real), float(phi.real)

def full_spin(U):
    det = np.linalg.det(U)
    U = U / np.sqrt(det)
    trace = np.trace(U)
    theta = -2 * np.arccos(np.clip(np.real(trace)/2, -1, 1))
    if np.isclose(theta, 0):
        return np.array([0, 0, 1]), 0
    nx = np.imag(U[1,0] + U[0,1]) / (2*np.sin(theta/2))
    ny = np.real(U[1,0] - U[0,1]) / (2*np.sin(theta/2))
    nz = np.imag(U[0,0] - U[1,1]) / (2*np.sin(theta/2))
    axis = np.array([nx, ny, nz])
    axis = axis / np.linalg.norm(axis)
    if axis[2] < 0:
        axis = -axis
        theta = -theta
    return axis.real, theta.real

def compute_slerp_path(gate, state):
    axis, theta_total = full_spin(gate)
    slerp_pts = []
    for t in np.linspace(0, 1, 50):
        theta_t = -t * theta_total
        n_dot_sigma = (
            axis[0] * pauli_x_gate +
            axis[1] * pauli_y_gate +
            axis[2] * pauli_z_gate)
        U_t = (np.cos(theta_t / 2) * np.eye(2) -
               1j * np.sin(theta_t / 2) * n_dot_sigma)
        state_t = U_t @ state
        state_t = state_t / np.linalg.norm(state_t)
        theta_b, phi_b = state_to_angles(state_t)
        x = np.sin(theta_b) * np.cos(phi_b)
        y = np.sin(theta_b) * np.sin(phi_b)
        z = np.cos(theta_b)
        slerp_pts.append([x, y, z])
    return np.array(slerp_pts).T

# ── Bloch sphere figure ───────────────────────────────────────────────────────
def Bloch_sphere(theta2, phi2):
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(len(u)), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1,
                             colorscale='viridis', showscale=False))

    x_arrow = np.sin(theta2) * np.cos(phi2)
    y_arrow = np.sin(theta2) * np.sin(phi2)
    z_arrow = np.cos(theta2)

    fig.add_trace(go.Scatter3d(
        x=[0, x_arrow], y=[0, y_arrow], z=[0, z_arrow],
        mode="lines", line=dict(color="red", width=5), name="Quantum State"))

    for i, historical_points in enumerate(st.session_state.slerp_history):
        fig.add_trace(go.Scatter3d(
            x=historical_points[0], y=historical_points[1], z=historical_points[2],
            mode="lines", line=dict(color='blue', width=4), opacity=0.3,
            name=f"Previous Path {i+1}", showlegend=False))

    equator_theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter3d(
        x=np.cos(equator_theta), y=np.sin(equator_theta),
        z=np.zeros_like(equator_theta),
        mode="lines", line=dict(color="gray", width=3, dash="dash"),
        name="Equator", showlegend=False))

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-1, 1],
        mode="lines", line=dict(color="black", width=4),
        opacity=0.2, name="Center Axis", showlegend=False))

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[1.3, -1],
        mode="text", text=["|0⟩", "|1⟩"],
        textposition="bottom center", showlegend=False,
        marker=dict(size=5, color="black")))

    fig.add_trace(go.Scatter3d(
        x=[x_arrow], y=[y_arrow], z=[z_arrow],
        mode="text", text=[f"({theta2:.2f}, {phi2:.2f})"],
        textposition="top center", showlegend=False,
        marker=dict(size=5, color="black")))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1, y=1, z=1))),
        title="Bloch Sphere",
        margin=dict(l=0, r=0, b=0, t=30),
        height=550)
    return fig

def prob_chart(theta):
    p0 = np.cos(theta/2) ** 2
    p1 = np.sin(theta/2) ** 2
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["|0⟩", "|1⟩"],
        y=[p0, p1],
        text=[f"{p0*100:.1f}%", f"{p1*100:.1f}%"],
        textposition="outside",
        marker_color=["#6929c4", "#1192e8"],
        width=0.4))
    fig.update_layout(
        title="Measurement Probabilities (Computational Basis)",
        yaxis=dict(range=[0, 1.2], tickformat=".0%",
                   title="Probability", gridcolor="lightgray"),
        xaxis=dict(title="Basis State"),
        height=300,
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white")
    return fig

# ── Sidebar: input mode ───────────────────────────────────────────────────────
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio(
    "Choose input type",
    ["Radians", "Complex Amplitudes", "Preset States"],
    index=["radians", "complex", "preset"].index(st.session_state.input_mode)
          if st.session_state.input_mode in ["radians", "complex", "preset"] else 0)

if input_mode == "Radians":
    st.session_state.input_mode = "radians"
elif input_mode == "Complex Amplitudes":
    st.session_state.input_mode = "complex"
else:
    st.session_state.input_mode = "preset"

st.sidebar.markdown("---")

# ── Sidebar: state input ──────────────────────────────────────────────────────
if st.session_state.input_mode == "radians":
    st.sidebar.subheader("Bloch Angles")
    theta_val = st.sidebar.number_input("θ (theta)", value=float(st.session_state.theta),
                                         min_value=0.0, max_value=float(np.pi), step=0.01, format="%.4f")
    phi_val   = st.sidebar.number_input("φ (phi)",   value=float(st.session_state.phi),
                                         min_value=0.0, max_value=float(2*np.pi), step=0.01, format="%.4f")
    if st.sidebar.button("Plot", type="primary"):
        st.session_state.theta = theta_val
        st.session_state.phi   = phi_val

elif st.session_state.input_mode == "complex":
    st.sidebar.subheader("Complex Amplitudes")
    state = angles_to_state(st.session_state.theta, st.session_state.phi)
    alpha_R = st.sidebar.number_input("α Real", value=round(float(np.real(state[0,0])), 6), step=0.01, format="%.6f")
    alpha_i = st.sidebar.number_input("α Imag", value=round(float(np.imag(state[0,0])), 6), step=0.01, format="%.6f")
    beta_R  = st.sidebar.number_input("β Real", value=round(float(np.real(state[1,0])), 6), step=0.01, format="%.6f")
    beta_i  = st.sidebar.number_input("β Imag", value=round(float(np.imag(state[1,0])), 6), step=0.01, format="%.6f")
    if st.sidebar.button("Plot", type="primary"):
        alpha = complex(alpha_R, alpha_i)
        beta  = complex(beta_R,  beta_i)
        s = np.array([[alpha], [beta]], dtype=complex)
        norm = np.linalg.norm(s)
        if norm < 1e-10:
            st.sidebar.error("State vector cannot be zero.")
        else:
            t, p = state_to_angles(s / norm)
            st.session_state.theta = t
            st.session_state.phi   = p

elif st.session_state.input_mode == "preset":
    st.sidebar.subheader("Preset States")
    preset_choice = st.sidebar.selectbox("Choose a preset", list(preset_states.keys()))
    if st.sidebar.button("Apply Preset", type="primary"):
        t, p = preset_states[preset_choice]
        st.session_state.theta = t
        st.session_state.phi   = p

# ── Sidebar: Circuit runner ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Circuit Runner")
circuit_start = st.sidebar.selectbox("Start from", list(preset_states.keys()), key="circuit_start")
circuit_str   = st.sidebar.text_input("Gates (comma-separated)",
                                       placeholder="e.g. H, X, S, T, Sdagger, Tdagger")
if st.sidebar.button("Run Circuit", type="primary"):
    if not circuit_str.strip():
        st.sidebar.error("⚠ Enter at least one gate.")
    else:
        gate_names = [g.strip() for g in circuit_str.split(",")]
        unknown = [g for g in gate_names if g not in circuit_inputs]
        if unknown:
            st.sidebar.error(f"⚠ Unknown gate(s): {', '.join(unknown)}")
        else:
            t, p = preset_states[circuit_start]
            state = angles_to_state(t, p)
            for name in gate_names:
                gate = circuit_inputs[name]
                path = compute_slerp_path(gate, state)
                st.session_state.slerp_history.append(path)
                if len(st.session_state.slerp_history) > 4:
                    st.session_state.slerp_history.pop(0)
                new_state = gate @ state
                state = new_state / np.linalg.norm(new_state)
            new_t, new_p = state_to_angles(state)
            st.session_state.theta = new_t
            st.session_state.phi   = new_p

if st.sidebar.button("Clear Path History"):
    st.session_state.slerp_history = []

# ── Main area: gate buttons ───────────────────────────────────────────────────
st.subheader("Apply Gate")
cols = st.columns(len(buttons))
for col, (name, gate) in zip(cols, buttons.items()):
    if col.button(name):
        state = angles_to_state(st.session_state.theta, st.session_state.phi)
        path = compute_slerp_path(gate, state)
        st.session_state.slerp_history.append(path)
        if len(st.session_state.slerp_history) > 4:
            st.session_state.slerp_history.pop(0)
        new_state = gate @ state
        new_state = new_state / np.linalg.norm(new_state)
        new_t, new_p = state_to_angles(new_state)
        st.session_state.theta = new_t
        st.session_state.phi   = new_p

# ── Main area: plots ──────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(Bloch_sphere(st.session_state.theta, st.session_state.phi),
                    use_container_width=True)
with col2:
    st.markdown("**Current State**")
    state = angles_to_state(st.session_state.theta, st.session_state.phi)
    st.latex(r"\theta = " + f"{st.session_state.theta:.4f}")
    st.latex(r"\phi = "   + f"{st.session_state.phi:.4f}")
    alpha = state[0, 0]
    beta  = state[1, 0]
    st.latex(r"\alpha = " + f"{np.real(alpha):.4f} + {np.imag(alpha):.4f}i")
    st.latex(r"\beta = "  + f"{np.real(beta):.4f} + {np.imag(beta):.4f}i")
    p0 = np.cos(st.session_state.theta/2) ** 2
    p1 = np.sin(st.session_state.theta/2) ** 2
    st.markdown(f"**P(|0⟩) = {p0*100:.1f}%**")
    st.markdown(f"**P(|1⟩) = {p1*100:.1f}%**")

st.markdown("**Measurement Probabilities**")
st.plotly_chart(prob_chart(st.session_state.theta), use_container_width=True)
