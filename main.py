import argparse
import datetime
import subprocess
import data
import models
from features import FeatureExtractor
import numpy as np

DATA_PATH = './data/2024_LoL_esports_match_data_from_OraclesElixir.csv'
FEATURE_NAME_FILE = './features.txt'
FEATURES_PATH = './checkpoints/checkpoint_20240901_234012/features.csv'

def parse_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, "%m-%d-%Y")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use MM-DD-YYYY format.")

def extract_features(start_date_str, end_date_str, label, feature_name_file):
    start_date = parse_date(start_date_str)
    end_date = parse_date(end_date_str)
    df = data.read_data(DATA_PATH)
    data.compute_features(df, start_date, end_date, label, features=data.get_features(feature_name_file))

def backtest_simulate(model_type, features_path, feature_name_file):
    X, Y = data.load_dataset(features_path, data.get_features(feature_name_file))
    if model_type == 'poisson':
        models.simulate_poisson(X, Y)

def run_inference(date_str, features_path, feature_name_file):
    date = parse_date(date_str)
    df = data.read_data(DATA_PATH)
    X, Y = data.load_dataset(features_path, data.get_features(feature_name_file))
    model = models.PoissonRegression(X, Y)
    while True:
        playername = input('Player Name: ')
        teamname = input('Team Name: ')
        opp_teamname = input('Opponent Team Name: ')
        position = input('Position: ')
        game = data.Input(
            date=date,
            playername=playername,
            teamname=teamname,
            opp_teamname=opp_teamname,
            position=position
        )
        fe = FeatureExtractor(df, game)
        input_nparr = np.array([[fe.extract(f) for f in data.get_features(feature_name_file)]])
        lam = model.predict(input_nparr)[0]
        lam = round(lam, 2)
        print(f'Prediction: {lam}')

def run_interactive():
    subprocess.run(["streamlit", "run", "interactive.py"], check=True)

def main():
    parser = argparse.ArgumentParser(description='Run betting model for LoL')
    subparsers = parser.add_subparsers(dest='command')

    # Extract features
    extract_parser = subparsers.add_parser('extract', help='Extract features to create a feature file')
    extract_parser.add_argument('start_date', type=parse_date, help='Start date for feature extraction in MM-DD-YYYY format')
    extract_parser.add_argument('end_date', type=parse_date, help='End date for feature extraction in MM-DD-YYYY format')
    extract_parser.add_argument('label', type=str, help='Label to use for feature extraction')
    extract_parser.add_argument('feature_name_file', type=str, help='Path to the file with newline separated features', default=FEATURE_NAME_FILE)

    # Backtest/Simulate model
    backtest_parser = subparsers.add_parser('backtest', help='Backtest or simulate a model')
    backtest_parser.add_argument('model_type', type=str, help='Type of model to use (e.g., poisson)')
    backtest_parser.add_argument('features_path', type=str, help='Path to the features dataset', default='./checkpoints/checkpoint_20240901_234012/features.csv')
    backtest_parser.add_argument('feature_name_file', type=str, help='Path to the file with newline separated features', default='./features.txt')

    # Run inference
    inference_parser = subparsers.add_parser('inference', help='Run a singular prediction (inference)')
    inference_parser.add_argument('date', type=parse_date, help='Date of the game for prediction in MM-DD-YYYY format')
    inference_parser.add_argument('features_path', type=str, help='Path to the features dataset', default=FEATURES_PATH)
    inference_parser.add_argument('feature_name_file', type=str, help='Path to the file with newline separated features', default=FEATURE_NAME_FILE)

    # Run interactive dashboard
    subparsers.add_parser('interactive', help='Run interactive panel')

    args = parser.parse_args()

    if args.command == 'extract':
        extract_features(args.start_date, args.end_date, args.label, args.feature_name_file)
    elif args.command == 'backtest':
        backtest_simulate(args.model_type, args.features_path, args.feature_name_file)
    elif args.command == 'inference':
        run_inference(args.date, args.features_path, args.feature_name_file)
    elif args.command == 'interactive':
        run_interactive()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
