import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode


def unbalanced_bagger_no_sampling(x, y, M=1, base_estimator=None, random_seed=None):
    if base_estimator is None:
        if random_seed is None:
            base_estimator = DecisionTreeClassifier()
        else:
            base_estimator = DecisionTreeClassifier(random_state=random_seed)

    base_learners = []
    for i in range(M):
        rnd = np.random.choice(x.shape[0], size=x.shape[0])
        x_bootstrap = x.iloc[rnd, :]
        y_bootstrap = y.iloc[rnd]
        base_estimator.fit(x_bootstrap, y_bootstrap)
        base_learners.append(base_estimator)

    def predict(x):
        predictions = np.zeros([x.shape[0], len(base_learners)])
        for i in range(len(base_learners)):
            predictions[:, i] = base_learners[i].predict(x)
        return mode(predictions, axis=1)[0]

    def predict_proba(x):
        n_classes = len(base_learners[0].classes_)
        proba = np.zeros([x.shape[0], n_classes])
        for i, base_learner in enumerate(base_learners):
            proba += base_learner.predict_proba(x)
        return proba / len(base_learners)

    return predict, predict_proba
