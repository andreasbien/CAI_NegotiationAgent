import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.metrics import mean_squared_error as mse
import csv
#domain 00
# actualWeights = {"issueA": 0.06667, "issueB": 0.04186, "issueC": 0.07844, "issueD": 0.70691, "issueE": 0.10612}
#domain 01
# actualWeights = {"issueA": 0.24962, "issueB": 0.24273,"issueC": 0.37965,"issueD": 0.128}

#domain 02
# actualWeights = {"issueA": 0.11155,
#       "issueB": 0.09718,
#       "issueC": 0.20828,
#       "issueD": 0.4526,
#       "issueE": 0.02944,
#       "issueF": 0.10095}
# actualWeights = {"issueA": 0.21776,"issueB": 0.05051,"issueC": 0.36,"issueD": 0.37173}

#domain 3
# actualWeights = {"issueA": 0.27463,
#       "issueB": 0.3612,
#       "issueC": 0.23751,
#       "issueD": 0.12666}

#domain 4
# actualWeights = {"issueA": 0.02571,
#       "issueB": 0.14084,
#       "issueC": 0.39066,
#       "issueD": 0.21482,
#       "issueE": 0.17048,
#       "issueF": 0.05749}

#domain 5
# actualWeights = {
#       "issueA": 0.09534,
#       "issueB": 0.00898,
#       "issueC": 0.59264,
#       "issueD": 0.30304
#     }
#domain 6
actualWeights = {
      "issueA": 0.02371,
      "issueB": 0.42255,
      "issueC": 0.33802,
      "issueD": 0.12867,
      "issueE": 0.08705
    }
df = pd.read_csv('data.csv')
unique_weights = df['window_size'].unique()

X = {}
Y = {}
plt.figure()
plt.title("agent_46 vs dreamteam109_agent domain06/profileA")
plt.ylabel("spearman rank correlation")
plt.xlabel("progress")
for window_size in unique_weights:
    X[window_size] = []
    Y[window_size] = []
    filtered_df = df[df['window_size'] == window_size]
    progresses = filtered_df['progress']

    for progress in progresses:
        progress_row = df[(df["progress"] == progress) & (df['window_size'] == window_size)].iloc[0]

        actual = []
        predicted = []
        for issue in actualWeights.keys():
            actual.append(actualWeights[issue])
            predicted.append(progress_row[issue])




        X[window_size].append(progress)

        #if all values are the same then we just score it at 0
        if min(predicted) > max(predicted) - 0.001:
            Y[window_size].append(0)
        else:
            res, p = sts.spearmanr(actual,predicted)
            Y[window_size].append(res)
    if window_size == 420:
        plt.plot(X[window_size], Y[window_size], label ="agent_46 frequency model")
    elif window_size ==1:
        plt.plot(X[window_size], Y[window_size], label="classical frequency model")
    else:
        plt.plot(X[window_size], Y[window_size], label="dfm window size =" + str(window_size))
plt.legend()
plt.show()







