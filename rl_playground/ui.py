from __future__ import annotations
import io
import base64
import time
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from rl_playground.agent import QLearningAgent
from rl_playground.environment import make_adapter
from rl_playground.deep import train_dqn  # requires: stable-baselines3, torch

# ==========================================================
# Page Setup — Fullscreen, Dark, Minimal Chrome
# ==========================================================
st.set_page_config(
    page_title="RL Playground",
    layout="wide",
    initial_sidebar_state="collapsed",  # default: keine Sidebar sichtbar
)
plt.style.use("dark_background")

# ==========================================================
# Global CSS — Deep Dark, 1-Screen, sichtbare Buttons
# ==========================================================
st.markdown(
    """
    <style>
      :root{
        --bg:#0b0f16; --bg2:#0f1520; --card:#121a27;
        --ink:#eaf0f7; --muted:#c3cbd6; --border:#273245;
        --accent:#4aa3ff; --accent-2:#ff5f57;
      }

      html, body, .stApp, [data-testid="stAppViewContainer"]{
        background:var(--bg)!important;
        color:var(--ink)!important;
        height:100%;
      }

      header, footer{visibility:hidden; height:0;}
      .block-container{padding:0!important; margin:0!important;}
      section.main>div{padding-top:0!important; padding-bottom:0!important;}
      body{overflow:hidden;} /* 1 Screen, kein Scroll */

      /* -------------------------
         Topbar
      ------------------------- */
      .topbar{
        display:flex;
        align-items:center;
        gap:10px;
        padding:10px 12px 6px;
      }

      /* ================================
         Panel-/Popover-Buttons (❔, ☰)
         Harter Fix: Idle NICHT hellgrau
      ================================ */
      [data-testid="stPopoverButton"] > button,
      [data-testid="stPopoverButton"] > div > button {
        background-color: #1a2434 !important;   /* dunkel im Idle */
        color: #eaf0f7 !important;              /* helle Schrift/Icon */
        border: 1px solid #33435d !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        height: 42px;
        padding: 8px 12px !important;
        background-image: none !important;
        box-shadow: none !important;
        filter: none !important;
        opacity: 1 !important;
      }

      [data-testid="stPopoverButton"] svg {
        color: currentColor !important;
        fill: currentColor !important;
      }

      [data-testid="stPopoverButton"] > button:hover,
      [data-testid="stPopoverButton"] > div > button:hover,
      [data-testid="stPopoverButton"][aria-expanded="true"] > button,
      [data-testid="stPopoverButton"][aria-expanded="true"] > div > button {
        background-color: var(--accent) !important;
        color: #0b1220 !important;
        border-color: var(--accent) !important;
        box-shadow: none !important;
      }

      [data-testid="stPopoverButton"] > button:focus-visible,
      [data-testid="stPopoverButton"] > div > button:focus-visible {
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
      }

      .btn-round [data-testid="stPopoverButton"] > button,
      .btn-round [data-testid="stPopoverButton"] > div > button {
        border-radius: 999px !important;
        width: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 !important;
      }

      /* -------------------------
         HUD-Chips
      ------------------------- */
      .hud{
        background:var(--card);
        border:1px solid var(--border);
        color:var(--ink);
        border-radius:10px;
        padding:6px 10px;
        font-weight:600;
        display:inline-block;
        font-size:.95rem;
      }

      /* -------------------------
         Responsive Env-Image
      ------------------------- */
      .env-wrap{
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
      }
      .env-wrap img{
        max-height:calc(100vh - 56px - 64px);
        height:auto;
        width:auto;
        max-width:96vw;
        display:block;
      }

      /* -------------------------
         Hinweis unten links
      ------------------------- */
      .hint{
        position:fixed;
        left:14px;
        bottom:20px;
        color:var(--muted);
        font-size:.9rem;
        z-index:5;
      }

      /* =========================================
         Universeller Fix für Streamlit-Buttons
         (betrifft auch Play/Train-Button)
      ========================================= */
      div.stButton{
        position: relative !important;
        display: inline-flex !important;
        isolation: isolate !important;
        border-radius: 12px !important;
      }

      div.stButton::before{
        content: "" !important;
        position: absolute !important;
        inset: 0 !important;
        background: #1a2434 !important;       /* Idle: dunkel */
        border: 1px solid #33435d !important;
        border-radius: 12px !important;
        z-index: 0 !important;
        box-shadow: none !important;
      }

      div.stButton > button{
        position: relative !important;
        z-index: 1 !important;
        background: transparent !important;
        background-image: none !important;
        color: #eaf0f7 !important;
        border: none !important;
        box-shadow: none !important;
        filter: none !important;
        opacity: 1 !important;
        height: 42px !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        -webkit-appearance: none !important;
        appearance: none !important;
        white-space: nowrap !important;
      }

      div.stButton:hover::before{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
      }

      div.stButton:hover > button{
        color: #0b1220 !important;
      }

      div.stButton > button:focus-visible{
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
      }

      /* ================================
   Hartes Override für Panel-/Popover-Buttons (❔, ☰)
   Erzwingt dunklen Idle-Hintergrund via ::before am Parent
   ================================ */

/* Parent vorbereiten */
[data-testid="stPopoverButton"]{
  position: relative !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  border-radius: 12px !important;
  isolation: isolate !important; /* sorgt dafür, dass ::before nicht "ausläuft" */
}

/* Der erzwungene Hintergrund */
[data-testid="stPopoverButton"]::before{
  content: "" !important;
  position: absolute !important;
  inset: 0 !important;                 /* füllt den gesamten Trigger */
  background: #1a2434 !important;      /* >>> dunkler Idle-Background <<< */
  border: 1px solid #33435d !important;
  border-radius: 12px !important;
  z-index: 0 !important;               /* liegt HINTER dem Button-Inhalt */
  box-shadow: none !important;
}

/* Inhalt (das echte <button>) vor den Hintergrund holen */
[data-testid="stPopoverButton"] > *,
[data-testid="stPopoverButton"] > button,
[data-testid="stPopoverButton"] > div > button{
  position: relative !important;
  z-index: 1 !important;
}

/* Den eigentlichen Button "neutralisieren", falls er hell sein will */
[data-testid="stPopoverButton"] > button,
[data-testid="stPopoverButton"] > div > button{
  background: transparent !important;   /* durchsichtig – Hintergrund kommt vom ::before */
  color: #eaf0f7 !important;            /* helle Schrift/Icon */
  border: none !important;
  box-shadow: none !important;
  filter: none !important;
  opacity: 1 !important;
  -webkit-appearance: none !important;
  appearance: none !important;
  height: 42px !important;
  padding: 8px 12px !important;
  border-radius: 12px !important;
}

/* Icons an Textfarbe koppeln */
[data-testid="stPopoverButton"] svg{
  color: currentColor !important;
  fill: currentColor !important;
}

/* Hover/Offen: nur den ::before-Hintergrund umfärben (Button bleibt transparent) */
[data-testid="stPopoverButton"]:hover::before,
[data-testid="stPopoverButton"][aria-expanded="true"]::before{
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* Beim Hover/Offen Textfarbe umschalten für Kontrast */
[data-testid="stPopoverButton"]:hover > *,
[data-testid="stPopoverButton"][aria-expanded="true"] > *{
  color: #0b1220 !important;  /* dunkler Text auf Akzent */
}

/* Tastatur-Fokus-Ring sichtbar, ohne graue Overlays */
[data-testid="stPopoverButton"] > button:focus-visible,
[data-testid="stPopoverButton"] > div > button:focus-visible{
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
  box-shadow: none !important;
}

/* Optional: runder ❔-Button links */
.btn-round [data-testid="stPopoverButton"]{
  border-radius: 999px !important;
}
.btn-round [data-testid="stPopoverButton"]::before{
  border-radius: 999px !important;
}
.btn-round [data-testid="stPopoverButton"] > button,
.btn-round [data-testid="stPopoverButton"] > div > button{
  border-radius: 999px !important;
  width: 42px !important;
  height: 42px !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# UI State
# ==========================================================
st.session_state.setdefault("algo", "Tabular Q-Learning")
st.session_state.setdefault("env_choice", "FrozenLake")
st.session_state.setdefault("seed", 42)

# Tabular defaults
st.session_state.setdefault("episodes", 600)
st.session_state.setdefault("max_steps", 120)
st.session_state.setdefault("alpha", 0.1)
st.session_state.setdefault("gamma", 0.95)
st.session_state.setdefault("epsilon", 0.2)
st.session_state.setdefault("epsilon_decay", 0.995)
st.session_state.setdefault("epsilon_min", 0.01)

st.session_state.setdefault("is_slippery", False)
st.session_state.setdefault("map_name", "4x4")
st.session_state.setdefault("live", True)
st.session_state.setdefault("animate_steps", True)
st.session_state.setdefault("update_every", 10)
st.session_state.setdefault("delay_ms", 120)

# ==========================================================
# One-screen Render Slot (verhindert Stacken/Scrollen)
# ==========================================================
ENV_SLOT = st.empty()

# ==========================================================
# Helpers
# ==========================================================
def env_goal_text(kind: str) -> str:
    return {
        "FrozenLake": "Goal: reach the goal tile (G) from the start (S) while avoiding holes (H).",
        "MountainCar": "Goal: drive up the right hill; gather momentum by moving back and forth.",
        "Energy Storage": "Goal: minimize energy cost by charging when prices are low and discharging when prices are high.",
        "CartPole": "Goal: balance the pole by applying left/right forces to keep it upright.",
    }.get(kind, "Goal: learn a policy that maximizes cumulative reward.")

def action_labels(kind: str) -> List[str]:
    return {
        "FrozenLake": ["0: LEFT", "1: DOWN", "2: RIGHT", "3: UP"],
        "MountainCar": ["0: PUSH LEFT", "1: COAST", "2: PUSH RIGHT"],
        "Energy Storage": ["0: DISCHARGE (-1)", "1: HOLD (0)", "2: CHARGE (+1)"],
        "CartPole": ["0: PUSH CART LEFT", "1: PUSH CART RIGHT"],
    }.get(kind, [])

def frozenlake_desc(map_name: str = "4x4") -> np.ndarray:
    if map_name == "4x4":
        lines = ["SFFF", "FHFH", "FFFH", "HFFG"]
    else:
        lines = [
            "SFFFFFFF","FFFFFFFF","FFFHFFFF","FFFFFHFF",
            "FFFHFFFF","FHHFFFHF","FHFFHFHF","FFFHFFFG",
        ]
    return np.array([list(row) for row in lines])

def fig_to_img_tag(fig, max_vh_offset: int = 126) -> str:
    """Render Matplotlib figure to base64 PNG and return an <img> tag.
    max_vh_offset ist bereits in CSS berücksichtigt; hier nur Bytes wrappen.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<div class="env-wrap"><img alt="env" src="data:image/png;base64,{b64}" /></div>'

def draw_frozenlake(map_name: str, agent_state: Optional[int] = None,
                    path: Optional[List[int]] = None, title: Optional[str] = None):
    """Responsive render via HTML <img> (max-height: 1 Screen)"""
    desc = frozenlake_desc(map_name)
    side = desc.shape[0]

    # relative Größen; tatsächliche Darstellung regelt CSS per max-height
    if side == 4:
        figsize = (6, 6)
        fsize = 12
        arrow = 0.17
        dot = 0.25
    else:
        figsize = (7.5, 7.5)
        fsize = 12
        arrow = 0.15
        dot = 0.22

    fig, ax = plt.subplots(figsize=figsize)
    color = {"S": (0.22, 0.45, 0.22), "G": (0.22, 0.45, 0.22), "F": (0.10, 0.12, 0.16), "H": (0.05, 0.06, 0.09)}
    grid = np.empty((side, side, 3))
    for i in range(side):
        for j in range(side):
            grid[i, j, :] = color.get(desc[i, j], (0.1, 0.1, 0.12))
    ax.imshow(grid, extent=[0, side, 0, side], origin="lower")
    for x in range(side + 1):
        ax.axvline(x, color="#2a2f3a", linewidth=0.7)
        ax.axhline(x, color="#2a2f3a", linewidth=0.7)
    for i in range(side):
        for j in range(side):
            ax.text(j + 0.5, i + 0.5, desc[i, j], ha="center", va="center",
                    fontsize=fsize, color="#eaf0f7" if desc[i, j] != "H" else "#9aa4b2")
    if path:
        for k in range(len(path) - 1):
            s = path[k]; i, j = divmod(s, side)
            s2 = path[k + 1]; i2, j2 = divmod(s2, side)
            ax.arrow(j + 0.5, i + 0.5, (j2 - j), (i2 - i),
                     head_width=arrow, head_length=arrow, length_includes_head=True, alpha=0.95)
    if agent_state is not None:
        i, j = divmod(agent_state, side)
        ax.add_patch(plt.Circle((j + 0.5, i + 0.5), dot, color="#ff5f57", alpha=0.98))
    ax.set(xticks=[], yticks=[])
    if title:
        ax.set_title(title, fontsize=12, color="#eaf0f7")

    # Wichtig: immer denselben Slot überschreiben (kein Stapeln)
    ENV_SLOT.markdown(fig_to_img_tag(fig), unsafe_allow_html=True)

def draw_mountaincar_snapshot(title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    xs = np.linspace(-1.2, 0.6, 200); ys = np.sin(3 * xs)
    ax.plot(xs, ys, linewidth=1.5)
    if title: ax.set_title(title, fontsize=12, color="#eaf0f7")
    ENV_SLOT.markdown(fig_to_img_tag(fig), unsafe_allow_html=True)

def draw_energy_snapshot(T: int, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(9, 3))
    t = np.arange(T); base = 0.5 + 0.5 * np.sin(2 * np.pi * t / max(1, len(t) // 2))
    ax.plot(t, base, linewidth=1.5)
    ax.set_xlabel("time"); ax.set_ylabel("price (demo)")
    if title: ax.set_title(title, fontsize=12, color="#eaf0f7")
    ENV_SLOT.markdown(fig_to_img_tag(fig), unsafe_allow_html=True)

def greedy_path_from_q(Q: np.ndarray, map_name: str = "4x4", max_steps: int = 200) -> List[int]:
    side = 4 if map_name == "4x4" else 8
    s = 0; path = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        i, j = divmod(s, side)
        if a == 0 and j > 0: j -= 1
        elif a == 1 and i < side - 1: i += 1
        elif a == 2 and j < side - 1: j += 1
        elif a == 3 and i > 0: i -= 1
        s = i * side + j
        path.append(s)
        if s == side * side - 1:
            break
    return path

# ==========================================================
# Topbar (Links Help-Popover, Mitte HUD, Rechts Panel-Popover)
# ==========================================================
c1, c2, c3 = st.columns([1, 6, 1])
with c1:
    st.markdown('<div class="topbar btn-round">', unsafe_allow_html=True)
    with st.popover("❔", use_container_width=False):
        st.markdown("### Quick Help")
        st.write("Default zeigt nur Environment & Play. Panels öffnen sich als Popover (Overlay) – bleiben auf einem Screen.")
        st.caption("FrozenLake greedy path basiert direkt auf Q (vereinfachte Visualisierung).")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(
        f'<div class="topbar"><span class="hud"><b>{st.session_state.env_choice}</b> — {env_goal_text(st.session_state.env_choice)}</span></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown('<div class="topbar btn-dark">', unsafe_allow_html=True)
    with st.popover("☰", use_container_width=False):
        st.markdown("### Parameters")
        st.session_state.algo = st.selectbox("Algorithm", ["Tabular Q-Learning", "DQN (SB3)"], index=0)
        if st.session_state.algo == "Tabular Q-Learning":
            st.session_state.env_choice = st.selectbox("Environment", ["FrozenLake", "MountainCar", "Energy Storage"], index=0)
        else:
            st.session_state.env_choice = st.selectbox("Environment", ["CartPole", "MountainCar"], index=0)
        st.session_state.seed = st.number_input("Random seed", value=int(st.session_state.seed), step=1)

        if st.session_state.algo == "Tabular Q-Learning":
            st.divider()
            st.markdown("**Training**")
            st.session_state.episodes = st.slider("Episodes", 50, 10000, int(st.session_state.episodes), 50)
            st.session_state.max_steps = st.slider("Max steps/episode", 10, 500, int(st.session_state.max_steps), 10)
            st.session_state.alpha = st.slider("Learning rate (α)", 0.01, 1.0, float(st.session_state.alpha), 0.01)
            st.session_state.gamma = st.slider("Discount (γ)", 0.0, 0.999, float(st.session_state.gamma), 0.001)
            st.session_state.epsilon = st.slider("Exploration (ε)", 0.0, 1.0, float(st.session_state.epsilon), 0.01)
            st.session_state.epsilon_decay = st.slider("Epsilon decay", 0.90, 1.00, float(st.session_state.epsilon_decay), 0.001)
            st.session_state.epsilon_min = st.slider("Epsilon min", 0.0, 0.5, float(st.session_state.epsilon_min), 0.01)

            st.divider()
            st.markdown("**Environment**")
            if st.session_state.env_choice == "FrozenLake":
                st.session_state.is_slippery = st.checkbox("Stochastic transitions (is_slippery)", value=bool(st.session_state.is_slippery))
                st.session_state.map_name = st.selectbox("Map", ["4x4", "8x8"], index=0 if st.session_state.map_name=="4x4" else 1)
            elif st.session_state.env_choice == "MountainCar":
                st.session_state.bins_pos = st.slider("Position bins", 6, 40, 18, 1)
                st.session_state.bins_vel = st.slider("Velocity bins", 4, 30, 14, 1)
            elif st.session_state.env_choice == "Energy Storage":
                st.session_state.storage_horizon = st.slider("Horizon (time steps)", 12, 240, 48, 6)
                st.session_state.storage_levels = st.slider("Capacity levels (SoC)", 3, 20, 6, 1)
                st.session_state.storage_volatility = st.slider("Price volatility", 0.0, 2.0, 0.5, 0.05)

            st.divider()
            st.markdown("**Animation**")
            st.session_state.live = st.checkbox("Animate training", value=bool(st.session_state.live))
            st.session_state.animate_steps = st.checkbox("Step-by-step (FrozenLake)", value=bool(st.session_state.animate_steps))
            st.session_state.update_every = st.slider("Update every N episodes", 1, 50, int(st.session_state.update_every), 1)
            st.session_state.delay_ms = st.slider("Delay per update (ms)", 0, 1500, int(st.session_state.delay_ms), 10)
        else:
            st.divider()
            st.markdown("**DQN (SB3)**")
            st.session_state.total_timesteps = st.slider("total_timesteps", 5_000, 200_000, 20_000, 5_000)
            st.session_state.dqn_lr = st.slider("learning_rate", 1e-5, 1e-2, 1e-3, format="%.5f")
            st.session_state.dqn_gamma = st.slider("gamma", 0.80, 0.999, 0.99, 0.001)
            st.session_state.target_update = st.slider("target_update_interval", 250, 5000, 1000, 250)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# Environment (initial snapshot, responsive)
# ==========================================================
env_choice = st.session_state.env_choice
if st.session_state.algo == "Tabular Q-Learning" and env_choice == "FrozenLake":
    draw_frozenlake(map_name=st.session_state.map_name, agent_state=None, path=None)
elif st.session_state.algo == "Tabular Q-Learning" and env_choice == "MountainCar":
    draw_mountaincar_snapshot("MountainCar — terrain")
elif st.session_state.algo == "Tabular Q-Learning" and env_choice == "Energy Storage":
    draw_energy_snapshot(48, "Energy Storage — price baseline")
else:
    st.info(f"{env_choice} — rewards will update during training.")

# ==========================================================
# Play Button + Hinweis (bleibt im 1 Screen)
# ==========================================================
h1, h2 = st.columns([1, 1])
with h1:
    st.markdown('<div class="hint">Default: only environment & play</div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="btn-dark" style="display:flex; justify-content:flex-end; padding:0 12px 12px;">', unsafe_allow_html=True)
    start_btn = st.button("▶ Play / Train")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# Training
# ==========================================================
def run_tabular(
    adapter,
    agent: QLearningAgent,
    *,
    episodes:int,
    max_steps:int,
    animate:bool,
    animate_steps:bool,
    update_every:int,
    delay_ms:int,
    env_choice:str,
    map_name:Optional[str],
    actions_sink=None,
    storage_horizon:Optional[int]=None,
    storage_levels:Optional[int]=None,
):
    rewards = []
    visits = np.zeros(adapter.n_states, dtype=np.int32)

    for ep in range(int(episodes)):
        s = adapter.reset()
        ep_return = 0.0

        if animate and animate_steps and env_choice == "FrozenLake":
            draw_frozenlake(map_name or "4x4", agent_state=s)

        for _ in range(int(max_steps)):
            a = agent.select_action(s)
            s_next, r, done, _ = adapter.step(a)
            agent.update(s, a, r, s_next, done)
            visits[s] += 1
            s = s_next
            ep_return += r

            if actions_sink is not None:
                actions_sink.write(f"Last action: {action_labels(env_choice)[a]}")

            if animate and animate_steps and env_choice == "FrozenLake":
                draw_frozenlake(map_name or "4x4", agent_state=s)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

            if done:
                break

            if delay_ms > 0 and animate and not animate_steps and (ep+1) % update_every == 0:
                time.sleep(max(0.05, delay_ms / 1000.0))

        rewards.append(ep_return)
        agent.decay_epsilon()

        if animate and ((ep + 1) % update_every == 0 or ep == episodes - 1):
            if env_choice == "FrozenLake":
                path = greedy_path_from_q(agent.Q, map_name or "4x4")
                draw_frozenlake(map_name or "4x4", path=path, title=f"Greedy path — episode {ep+1}/{episodes}")
            elif env_choice == "MountainCar":
                draw_mountaincar_snapshot(f"MountainCar — after episode {ep+1}")
            elif env_choice == "Energy Storage":
                T = storage_horizon or 48
                draw_energy_snapshot(T, f"Energy Storage — after episode {ep+1}")
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    return np.asarray(rewards), visits, agent

# Diagnostics anchor (nur wenn gebraucht)
actions_sink = st.empty() if start_btn else None

# Start training
if start_btn:
    # Optional: kurzer Start-Hinweis im Slot
    ENV_SLOT.markdown("<div class='env-wrap'><div class='hud'>Training startet…</div></div>", unsafe_allow_html=True)

    if st.session_state.algo == "Tabular Q-Learning":
        kwargs = {}
        if env_choice == "FrozenLake":
            kwargs.update(dict(is_slippery=bool(st.session_state.is_slippery), map_name=str(st.session_state.map_name)))
        elif env_choice == "MountainCar":
            kwargs.update(dict(bins_pos=int(st.session_state.get("bins_pos", 18)),
                               bins_vel=int(st.session_state.get("bins_vel", 14))))
        elif env_choice == "Energy Storage":
            storage_h = int(st.session_state.get("storage_horizon", 48))
            storage_l = int(st.session_state.get("storage_levels", 6))
            kwargs.update(dict(storage_horizon=storage_h, storage_levels=storage_l,
                               storage_volatility=float(st.session_state.get("storage_volatility", 0.5))))
        adapter = make_adapter(env_choice, seed=int(st.session_state.seed), **kwargs)

        agent = QLearningAgent(
            n_states=adapter.n_states, n_actions=adapter.n_actions,
            alpha=float(st.session_state.alpha), gamma=float(st.session_state.gamma),
            epsilon=float(st.session_state.epsilon), epsilon_min=float(st.session_state.epsilon_min),
            epsilon_decay=float(st.session_state.epsilon_decay),
        )

        rewards, visits, agent = run_tabular(
            adapter, agent,
            episodes=int(st.session_state.episodes), max_steps=int(st.session_state.max_steps),
            animate=bool(st.session_state.live), animate_steps=bool(st.session_state.animate_steps),
            update_every=int(st.session_state.update_every), delay_ms=int(st.session_state.delay_ms),
            env_choice=env_choice, map_name=str(st.session_state.map_name) if env_choice == "FrozenLake" else None,
            actions_sink=actions_sink,
            storage_horizon=st.session_state.get("storage_horizon", None) if env_choice == "Energy Storage" else None,
            storage_levels=st.session_state.get("storage_levels", None) if env_choice == "Energy Storage" else None,
        )

        # Kleine Diagnostics als Popover (damit 1 Screen bleibt)
        with st.popover("📈 Diagnostics", use_container_width=False):
            st.write("Q-max / Rewards / Visits")
            qmax = np.max(agent.Q, axis=1)
            st.line_chart(qmax)
            st.line_chart(rewards)
            st.write("Visits sample:", visits[: min(20, len(visits))])

    else:
        env_id = "CartPole-v1" if env_choice == "CartPole" else "MountainCar-v0"
        try:
            model, ep_rewards = train_dqn(
                env_id=env_id,
                total_timesteps=int(st.session_state.total_timesteps),
                seed=int(st.session_state.seed),
                learning_rate=float(st.session_state.dqn_lr),
                gamma=float(st.session_state.dqn_gamma),
                target_update_interval=int(st.session_state.target_update),
            )
        except RuntimeError as e:
            st.error(str(e))
        else:
            with st.popover("📈 Diagnostics", use_container_width=False):
                st.write("Episode rewards")
                st.line_chart(ep_rewards)
            st.success("Training complete.")
