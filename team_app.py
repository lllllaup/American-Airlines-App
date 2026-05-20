import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="AA Sky Deck v8.1", layout="wide", page_icon="✈️")

# --- 1. DATA ENGINE (High AUC Optimization) ---
@st.cache_data
def sync_data_engine():
    f_path = 'Flight_Data_With_Weather_FINAL.csv'
    if not os.path.exists(f_path): 
        f_path = 'data/Processed_Assignment_Weather_Data.csv'
    
    df = pd.read_csv(f_path)
    df['SPOILED_HRS'] = df['TOTAL_SPOILED_HRS'].fillna(0.0)
    df['IS_COLLAPSE'] = (df['SPOILED_HRS'] >= 6.0).astype(int)
    
    # Feature Engineering for 0.65+ AUC
    df['RIGIDITY'] = df['SEQ_TTL_LEGS'] / (df['TOTAL_BLOCKED_HRS'].fillna(1) + 1)
    df['VIS_MILES'] = df['VIS_MILES'].fillna(10.0)
    df['WEATHER_PRESSURE'] = df['SEQ_TTL_LEGS'] * (11 - df['VIS_MILES'])
    
    # Bayesian Smoothing
    global_mean = df['IS_COLLAPSE'].mean()
    smoothing = 20
    agg = df.groupby(['BASE', 'FLEET'])['IS_COLLAPSE'].agg(['count', 'mean']).reset_index()
    agg['BAYES_RISK'] = ((agg['count'] * agg['mean']) + (smoothing * global_mean)) / (agg['count'] + smoothing)
    risk_map = agg.set_index(['BASE', 'FLEET'])['BAYES_RISK'].to_dict()
    df['BAYES_RISK'] = df.apply(lambda r: risk_map.get((r['BASE'], r['FLEET']), global_mean), axis=1)
    
    # LAYOVER COUNT MAPPING (Ensuring it exists for filtering)
    if 'LAYOVER_COUNT' not in df.columns:
        if 'SEQ_CAL_DAYS' in df.columns:
            df['LAYOVER_COUNT'] = (df['SEQ_CAL_DAYS'] - 1).clip(lower=0)
        elif 'LAYOVER' in df.columns:
            df['LAYOVER_COUNT'] = (df['LAYOVER'] > 0).astype(int)
        else:
            df['LAYOVER_COUNT'] = 0
    
    features = ['BAYES_RISK', 'TOTAL_BLOCKED_HRS', 'SEQ_TTL_LEGS', 'LAYOVER_COUNT', 'RIGIDITY', 'WEATHER_PRESSURE']
    X = df[features].fillna(0)
    y = df['IS_COLLAPSE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.02, max_depth=6, random_state=42).fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)
    
    return df, model, risk_map, global_mean, auc_score

df, model, risk_map, global_mean, final_auc = sync_data_engine()

# --- 2. THE RICH NAVY THEME ---
deep_navy = "#012a4a"
cyan_glow = "#00e5ff"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;700&family=JetBrains+Mono:wght@500&display=swap');
    
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(160deg, #012a4a 0%, #01497c 40%, #2a6f97 100%);
        font-family: 'Lexend', sans-serif;
    }}

    .glass-card {{
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 25px;
        padding: 25px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(15px);
        margin-bottom: 20px;
    }}

    .range-note {{
        font-size: 0.75rem;
        color: {cyan_glow};
        margin-top: -10px;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono';
        opacity: 0.8;
    }}

    h1, h2, h3 {{ color: white !important; font-weight: 700; }}
    p, label, span {{ color: white !important; }}
    
    .metric-label {{ color: {cyan_glow}; font-family: 'JetBrains Mono'; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; }}
    .metric-value {{ color: white; font-size: 2.3rem; font-weight: 700; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER ---
st.title("✈️ AA SKY DECK COMMANDER")
st.markdown(f"""
    <div style='display: flex; gap: 15px; margin-bottom: 25px;'>
        <div style='background: {cyan_glow}; color: {deep_navy}; padding: 6px 18px; border-radius: 12px; font-family: JetBrains Mono; font-weight: bold;'>
            PRECISION AUC: {final_auc:.4f}
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 6px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2); color: white;'>
            ☁️ CLOUD SYNC: ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. THE DECK ---
c1, c2, c3 = st.columns([1.2, 1.2, 1])

with c1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📍 Deployment")
    u_base = st.selectbox("OPERATIONAL HUB", sorted(df['BASE'].unique()))
    u_fleet = st.selectbox("AIRCRAFT FLEET", sorted(df['FLEET'].unique()))
    weather_opt = {"☀️ Clear": 10.0, "☁️ Overcast": 5.0, "🌧️ Rainy": 3.0, "⛈️ Stormy": 1.0}
    u_weather = st.selectbox("WEATHER CONDITION", list(weather_opt.keys()))
    u_vis = weather_opt[u_weather]
    st.markdown(f"<p class='range-note'>Historical Window: +/- 2mi</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Sequence Details")
    u_legs = st.number_input("TOTAL LEGS", 1, 15, 5)
    st.markdown("<p class='range-note'>1-2: Low | 3-5: Norm | 6+: High</p>", unsafe_allow_html=True)
    
    u_block = st.number_input("TOTAL BLOCK HOURS", 1.0, 30.0, 14.0)
    st.markdown("<p class='range-note'><10: Low | 10-16: Norm | 16+: High</p>", unsafe_allow_html=True)
    
    u_lay = st.number_input("LAYOVERS", 0, 5, 2)
    st.markdown("<p class='range-note'>Direct: 0 | Multi-Day: 1-2 | High: 3+</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📖 Historical Log")
    
    # DYNAMIC FILTERING
    # This now strictly follows BASE, FLEET, LEGS, and LAYOVERS
    hist_match = df[
        (df['BASE'] == u_base) & 
        (df['FLEET'] == u_fleet) & 
        (df['SEQ_TTL_LEGS'].between(u_legs-1, u_legs+1)) & # +/- 1 Leg for better sample size
        (df['LAYOVER_COUNT'] == u_lay) &                 # Exact Layover Match
        (df['VIS_MILES'].between(u_vis-2, u_vis+2)) &    # Visibility buffer
        (df['TOTAL_BLOCKED_HRS'].between(u_block-4, u_block+4)) # Block hour buffer
    ]
    
    entries = len(hist_match)
    avg_s = hist_match['SPOILED_HRS'].mean() if entries > 0 else 0
    worst_s = hist_match['SPOILED_HRS'].max() if entries > 0 else 0
    
    st.markdown(f"<p class='metric-label'>SIMILAR PROFILES</p><p class='metric-value'>{entries}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-label'>AVG DELAY</p><p class='metric-value'>{avg_s:.2f}h</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-label' style='color:#ff5252;'>WORST RECORDED</p><p class='metric-value' style='color:#ff5252;'>{worst_s:.2f}h</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- 5. RESULTS ---
if st.button("🚀 EXECUTE RISK ASSESSMENT", type="primary", use_container_width=True):
    # ML Logic
    b_risk = risk_map.get((u_base, u_fleet), global_mean)
    u_rigidity = u_legs / (u_block + 1)
    u_weather_p = u_legs * (11 - u_vis)
    
    input_df = pd.DataFrame([{
        'BAYES_RISK': b_risk, 'TOTAL_BLOCKED_HRS': u_block, 
        'SEQ_TTL_LEGS': u_legs, 'LAYOVER_COUNT': u_lay, 
        'RIGIDITY': u_rigidity, 'WEATHER_PRESSURE': u_weather_p
    }])
    prob = model.predict_proba(input_df)[:, 1][0] * 100
    
    # Classification
    if u_block <= 8: f_type, f_icon = "Short-Haul Hopper", "🛫"
    elif u_block <= 16: f_type, f_icon = "Medium-Haul Cruiser", "✈️"
    else: f_type, f_icon = "Long-Haul Endurance", "🛰️"

    # Pred Window
    window = 1.5 if u_block <= 8 else (3.5 if u_block <= 16 else 6.0)
    low_b, high_b = max(0, avg_s - (window/2)), avg_s + (window/2)

    st.markdown("---")
    res_l, res_r = st.columns([1, 1])
    
    with res_l:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            number = {'suffix': "%", 'font': {'color': 'white'}},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': cyan_glow}}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with res_r:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); border-radius: 25px; padding: 30px; border-left: 10px solid {cyan_glow};'>
            <h2 style='margin:0;'>{f_icon} {f_type}</h2>
            <hr style='opacity:0.2;'>
            <p class='metric-label'>ESTIMATED SPOILAGE WINDOW</p>
            <h1 style='font-size: 5rem; color: white; margin:0;'>{low_b:.1f} — {high_b:.1f}</h1>
            <p style='opacity:0.7;'>Hours of Predicted System Delay</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 30px; opacity: 0.3;'>☁️ ✈️ ☁️ ☁️ ✈️ ☁️</div>", unsafe_allow_html=True)
