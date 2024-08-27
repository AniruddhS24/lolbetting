class LabelExtractor:
    def __init__(self):
        self.name2func = {
            'kills': self.kills,
            'winlose': self.winlose,
        }
    
    def extract(self, df_row, label_name):
        return self.name2func[label_name](df_row)
    
    def kills(self, league_data_row):
        return league_data_row['kills']

    def winlose(self, league_data_row):
        return league_data_row['result']