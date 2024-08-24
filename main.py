import data
from features import FeatureExtractor

if __name__ == '__main__':
    player_name = 'CrabLord'
    game_date = '6/24/24 16:57'
    features = ['agt', 'apg', 'csdiffat10']

    df, game = data.get_game(player_name, game_date)
    fe = FeatureExtractor(df, game)
    for feat in features:
        print(f'{feat}: {fe.extract(feat):.2f}')
    
    # EXAMPLE IF YOU WANT TO TRAIN A MODEL:
    
    # x, y = make_dataset(df, datetime(2024, 6, 23), datetime(2024, 6, 25), [features.agt], labels.kills)
    # print(y)