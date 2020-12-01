import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

def unbalanced_bagger_no_sampling(x, y, M = 1):
    base_learners = []
    for i in range(M):
        rnd = np.random.choice(x.shape[0], size=x.shape[0])
        x_bootstrap = x.iloc[rnd,:]
        y_bootstrap = y.iloc[rnd]
        clf = DecisionTreeClassifier()
        clf.fit(x_bootstrap,y_bootstrap)
        base_learners.append(clf)

    def predict(x):
        predictions = np.zeros([x.shape[0], len(base_learners)])
        for i in range(len(base_learners)):
            predictions[:,i] = base_learners[i].predict(x)
        return mode(predictions, axis=1)[0]
    return predict