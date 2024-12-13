import numpy as np

class PrimalSVM:
    def __init__(self, C=1.0, init_learning_rate=0.01, lr_func = 'a', max_epochs=100):
        self.C = C
        self.init_learning_rate = init_learning_rate
        self.lr_func = lr_func
        self.max_epochs = max_epochs
        self.w = None
        self.b = None

    def train(self, X, y):
        self.num_examples, self.dimensions = X.shape
        self.w = np.zeros(self.dimensions)
        self.b = 0

        init_learning_rate = self.init_learning_rate
        learning_rate = 0

        y = 2 * y - 1

        for epoch in range(self.max_epochs):
            if(self.lr_func == 'a'):
                learning_rate = calc_learning_rate_a(init_learning_rate, epoch)
            if(self.lr_func == 'b'):
                learning_rate = calc_learning_rate_b(init_learning_rate, epoch)
            for i in range(self.num_examples):
                if y[i] * (np.dot(self.w, X[i]) + self.b) < 1:
                    # Misclassified
                    self.w -= learning_rate * (2 * self.C * self.w - np.dot(y[i], X[i]))
                    self.b -= learning_rate * (-y[i])
                else:
                    # Correctly classified
                    self.w -= learning_rate * (2 * self.C * self.w)

    # def predict(self, X):
    #     decision_function = np.dot(X, self.w) + self.b
    #     return np.sign(decision_function)

    # def accuracy(self, X, y):
    #     predictions = self.predict(X)
    #     return np.mean(predictions == y)
    
    def predict_proba(self, X):
        """Averages predictions (in this case, signed distances) across all base learners."""
        base_learner_decision_values = []

        for svm in self.base_learners:
            decision_values = np.dot(X, svm.w) + svm.b
            base_learner_decision_values.append(decision_values)

        # Average decision values
        avg_decision_values = np.mean(base_learner_decision_values, axis=0)

        # Apply sigmoid to convert decision values to probabilities
        probs = 1 / (1 + np.exp(-avg_decision_values))
        return np.column_stack((1 - probs, probs))
    
    def predict(self, X):
        """Predicts class labels based on averaged decision values."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def accuracy(self, X, y):
        """Calculates accuracy of predictions."""
        predictions = self.predict(X)
        return np.mean(predictions == (y > 0))  # Convert {-1, 1} to {0, 1} for comparison
    
def calc_learning_rate_a(learning_rate, epoch):
        return learning_rate / (1 + ((learning_rate / 10) * epoch))
    
def calc_learning_rate_b(learning_rate, epoch):
        return learning_rate / (1 + epoch)
