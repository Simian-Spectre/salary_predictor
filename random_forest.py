import numpy as np
from decision_tree import DecisionTree

class RandomForestClassifier():
    """
    Random Forest Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, n_base_learner=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0, \
                 numb_of_features_splitting=None, bootstrap_sample_size=None) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size
        self.tree_oob_errors = []

    def _create_bootstrap_samples(self, X, Y) -> tuple:
        """
        Creates bootstrap samples for each base learner
        """
        bootstrap_samples_X = []
        bootstrap_samples_Y = []
        oob_indices = []

        for i in range(self.n_base_learner):
            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = X.shape[0]
            
            sampled_idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            bootstrap_samples_X.append(X[sampled_idx])
            bootstrap_samples_Y.append(Y[sampled_idx])

            oob_idx = np.setdiff1d(range(X.shape[0]), sampled_idx)
            oob_indices.append(oob_idx)

        return bootstrap_samples_X, bootstrap_samples_Y, oob_indices

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """Trains the model with given X and Y datasets"""
        bootstrap_samples_X, bootstrap_samples_Y, oob_indices = self._create_bootstrap_samples(X_train, Y_train)

        self.oob_indices = oob_indices
        self.base_learner_list = []

        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTree(max_depth=self.max_depth, 
                                        min_samples_leaf=self.min_samples_leaf,
                                        min_information_gain=self.min_information_gain, 
                                        numb_of_features_splitting=self.numb_of_features_splitting)
            
            base_learner.train(bootstrap_samples_X[base_learner_idx], bootstrap_samples_Y[base_learner_idx])
            self.base_learner_list.append(base_learner)

            # Calculate OOB error for the current tree
            oob_idx = self.oob_indices[base_learner_idx]
            if len(oob_idx) > 0:
                oob_predictions = base_learner.predict(X_train[oob_idx])
                oob_error = 1 - np.mean(oob_predictions == Y_train[oob_idx])
            else:
                oob_error = 1.0 
            self.tree_oob_errors.append(oob_error)

        # Calculate feature importance
        self.feature_importances = self._calculate_rf_feature_importance(self.base_learner_list)

    def train_with_oob(self, X_train, Y_train):
        """Train the model and compute OOB error."""
        self.train(X_train, Y_train)
        oob_error = self.calculate_oob_error(X_train, Y_train)
        print(f"OOB Error: {oob_error:.4f}")

    def calculate_oob_error(self, X: np.array, Y: np.array) -> float:
        """
        Calculates the Out-of-Bag (OOB) error.
        """
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, self.n_base_learner))

        for i, base_learner in enumerate(self.base_learner_list):
            oob_idx = self.oob_indices[i]

            if len(oob_idx) > 0:
                # Predict probabilities for OOB samples
                pred_probs = base_learner.predict_proba(X[oob_idx])
                oob_predictions[oob_idx, i] = np.argmax(pred_probs, axis=1)

        # Aggregate predictions by majority voting
        aggregated_predictions = []
        for sample_idx in range(n_samples):
            # Ignore samples without OOB predictions
            nonzero_predictions = oob_predictions[sample_idx][oob_predictions[sample_idx] != 0]
            if len(nonzero_predictions) > 0:
                majority_vote = np.argmax(np.bincount(nonzero_predictions.astype(int)))
                aggregated_predictions.append(majority_vote)
            else:
                aggregated_predictions.append(-1)

        # Calculate OOB error
        valid_indices = np.where(np.array(aggregated_predictions) != -1)[0]
        oob_error = 1 - np.mean(
            np.array(aggregated_predictions)[valid_indices] == Y[valid_indices]
        )
        return oob_error

    def calculate_tree_weights(self):
        """
        Converts OOB errors into weights for each tree.
        Trees with lower OOB errors get higher weights.
        """
        almost_zero = 1e-10
        oob_errors = np.array(self.tree_oob_errors)
        tree_weights = 1 / (oob_errors + almost_zero)
        normalized_weights = tree_weights / np.sum(tree_weights)
        return normalized_weights


    def _predict_proba_w_base_learners(self,  X_set: np.array) -> list:
        """
        Creates list of predictions for all base learners
        """
        pred_prob_list = []
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(X_set))

        return pred_prob_list

    def predict_proba(self, X_set: np.array) -> list:
        """Returns the predicted probs for a given data set"""

        pred_probs = []
        base_learners_pred_probs = self._predict_proba_w_base_learners(X_set)

        # Get weights for each tree
        tree_weights = self.calculate_tree_weights()

        # Average the predicted probabilities of base learners
        for obs in range(X_set.shape[0]):
            base_learner_probs_for_obs = [a[obs] for a in base_learners_pred_probs]
            # Calculate the average for each index
            obs_weighted_pred_probs = np.average(base_learner_probs_for_obs, axis=0, weights=tree_weights)
            pred_probs.append(obs_weighted_pred_probs)

        return pred_probs

    def predict(self, X_set: np.array, return_probabilities=False) -> np.array:
        """Returns either the predicted probabilities or the predicted labels."""
        pred_probs = self.predict_proba(X_set)

        if return_probabilities:
            return pred_probs
        else:
            preds = np.argmax(pred_probs, axis=1)
            return preds
    
    def _calculate_rf_feature_importance(self, base_learners):
        """Calcalates the average feature importance of the base learners"""
        feature_importance_dict_list = []
        for base_learner in base_learners:
            feature_importance_dict_list.append(base_learner.feature_importances)

        feature_importance_list = [list(x.values()) for x in feature_importance_dict_list]
        average_feature_importance = np.mean(feature_importance_list, axis=0)

        return average_feature_importance