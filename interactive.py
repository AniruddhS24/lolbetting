import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import data
import models
from features import FeatureExtractor

st.set_page_config(page_title="Poisson Distribution CDF", layout="centered")

DATA_PATH = '2024_LoL_esports_match_data_from_OraclesElixir.csv'
FEATURES_PATH = './features.csv'
FEATURES = ['agt', 'apg', 'csdiffat10', 'opp_gives_up_kills']

# Placeholder function to load data
@st.cache_data
def load_data():
    df = data.read_data(DATA_PATH)
    if not FEATURES_PATH:
        # logger.info(f'Extracting features: {FEATURES}')
        X, Y = data.make_dataset(df, datetime(2024, 7, 10), datetime.now(), FEATURES, 'kills', './features.csv')
    else:
        # logger.info(f'Loading precomputed features: {FEATURES}')
        X, Y = data.load_dataset(FEATURES_PATH, FEATURES, 'kills')
    return df, X, Y

# Placeholder function to train a model
@st.cache_resource
def train_model(X, Y):
    # logger.info('Running Montecarlo simulations to fit model...')
    model = models.PoissonRegression(X, Y)
    return model

# Function to perform inference using the trained model
def inference(df, game):
    # Replace this logic with your actual inference logic
    fe = FeatureExtractor(df, game)
    lam = model.predict(np.array([[fe.extract(f) for f in FEATURES]]))[0]
    lam = round(lam, 2)
    return lam

# Load data and train model once, when the app starts
df, X, Y = load_data()
model = train_model(X, Y)

st.title('LoL Modeling')
_players = list(df['playername'].unique())
_teams = list(df['teamname'].unique())
# User input fields
date = st.date_input('Game Date')
time = st.time_input('Game Time')

playername = st.selectbox('Player Name', _players)
teamname = st.selectbox('Team', _teams)
opponent = st.selectbox('Opponent Team', _teams)
position = st.selectbox('Select Position', ['top', 'mid', 'bot', 'jng', 'sup'])
num_games = st.number_input('Number of Games', step=1, value=1)
x_val = st.slider('Line:', min_value=0.0, max_value=20.0, value=0.0, step=0.5)

if 'lambda_poisson' not in st.session_state:
    st.session_state.lambda_poisson = 0

# Enter button
if st.button('Enter'):
    datetime_combined = datetime.combine(date, time)
    # Run inference to calculate lambda based on inputs
    game = data.Input(
        date=datetime_combined,
        playername=playername,
        teamname=teamname,
        opp_teamname=opponent,
        position=position
    )
    st.session_state.lambda_poisson = inference(df, game)*num_games

# User input for x after lambda is calculated

lambda_poisson = st.session_state.lambda_poisson
# Calculate CDF
cdf_val = poisson.cdf(x_val, lambda_poisson)

# Display the calculated lambda and CDF value
# st.markdown(f"<h3 style='text-align: center;'>Calculated λ = {lambda_poisson:.2f}</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>P(kills < {x_val}) = {cdf_val*100:.2f}%</h3>", unsafe_allow_html=True)

# Prepare data for Altair plot
x = np.arange(0, 20)
pmf = poisson.pmf(x, lambda_poisson)
data = pd.DataFrame({'X': x, 'PMF': pmf})
highlight_data = pd.DataFrame({'X': np.arange(0, x_val + 1), 'PMF': poisson.pmf(np.arange(0, x_val + 1), lambda_poisson)})

# Base plot
base = alt.Chart(data).mark_line(point=True).encode(
    x=alt.X('X:Q', axis=alt.Axis(title='X')),
    y=alt.Y('PMF:Q', axis=alt.Axis(title='PMF'))
).properties(
    width=600,
    height=400,
    title=f'Poisson Kill Distribution (λ = {lambda_poisson})'
)

highlight = alt.Chart(pd.DataFrame({'X': np.append(np.arange(0, int(np.floor(x_val)) + 1), x_val), 'PMF': np.append(poisson.pmf(np.arange(0, int(np.floor(x_val)) + 1), lambda_poisson), poisson.pmf(int(np.floor(x_val)), lambda_poisson) + (x_val - np.floor(x_val)) * (poisson.pmf(int(np.ceil(x_val)), lambda_poisson) - poisson.pmf(int(np.floor(x_val)), lambda_poisson)))})).mark_area(opacity=0.3, color='red').encode(x='X:Q', y='PMF:Q')
vertical_line = alt.Chart(pd.DataFrame({'X': [x_val], 'PMF': [0], 'PMF_end': [poisson.pmf(int(np.floor(x_val)), lambda_poisson) + (x_val - np.floor(x_val)) * (poisson.pmf(int(np.ceil(x_val)), lambda_poisson) - poisson.pmf(int(np.floor(x_val)), lambda_poisson))]})).mark_rule(color='red', strokeDash=[5, 5]).encode(x='X:Q', y='PMF:Q', y2='PMF_end:Q')
final_chart = base + highlight + vertical_line

st.altair_chart(final_chart)
