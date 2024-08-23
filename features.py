def agt(games):
   return list(games.iloc[-5:]['gamelength'])

def apg(games):
    return list(games.iloc[-5:]['assists'])

def deaths(games):
    return list(games.iloc[-5:]['deaths'])

def csdiffat10(games):
    return list(games.iloc[-5:]['csdiffat10'])

def csdiffat15(games):
    return list(games.iloc[-5:]['csdiffat15'])

def csdiffat20(games):
    return list(games.iloc[-5:]['csdiffat20'])

def csdiffat25(games):
    return list(games.iloc[-5:]['csdiffat25'])