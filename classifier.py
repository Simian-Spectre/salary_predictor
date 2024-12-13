from decision_tree import DecisionTree
from random_forest import RandomForestClassifier
from ada_boost import AdaBoostClassifier
from SVM import PrimalSVM
import numpy as np
import random
import csv

def read_csv(filename):
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        is_train = 'income>50K' in reader.fieldnames            
        people = []
        labels = []
        for row in reader:
            attributes = []

            if is_train:
                labels.append(int(row['income>50K']))
            else:
                attributes.append(int(row['ID']))

            attributes.append(int(row['age']))
            attributes.append(quant_workclass(row['workclass']))
            attributes.append(int(row['fnlwgt']))
            attributes.append(quant_education(row['education']))
            attributes.append(int(row['education.num']))
            attributes.append(quant_marital_status(row['marital.status']))
            attributes.append(quant_occupation(row['occupation']))
            attributes.append(quant_relationship(row['relationship']))
            attributes.append(quant_race(row['race']))
            attributes.append(quant_sex(row['sex']))
            attributes.append(int(row['capital.gain']))
            attributes.append(int(row['capital.loss']))
            attributes.append(int(row['hours.per.week']))
            attributes.append(quant_native_country(row['native.country']))

            people.append(attributes)
    return people, labels

def shuffle_data(data, labels):
    # Combine data and labels into a list of tuples
    combined = list(zip(data, labels))
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Unzip the shuffled data back into separate lists
    shuffled_data, shuffled_labels = zip(*combined)
    
    return list(shuffled_data), list(shuffled_labels)

def split_data(data, labels, split=0.9):
    split_index = int(len(data) * split)

    # Split into training and testing sets
    train_data = data[:split_index]
    train_labels = labels[:split_index]
    test_data = data[split_index:]
    test_labels = labels[split_index:]
    
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def quant_workclass(workclass):
    quantifier = [
		"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
		"Local-gov", "State-gov", "Without-pay", "Never-worked"
		]
    try:
        return quantifier.index(workclass)
    except ValueError:
        return -1

def quant_education(education):
    quantifier = [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", 
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", 
        "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
    ]
    try:
        return quantifier.index(education)
    except ValueError:
        return -1

def quant_marital_status(marital_status):
    quantifier = [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", 
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ]
    try:
        return quantifier.index(marital_status)
    except ValueError:
        return -1

def quant_occupation(occupation):
    quantifier = [
        "Tech-support", "Craft-repair", "Other-service", "Sales", 
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
        "Transport-moving", "Priv-house-serv", "Protective-serv", 
        "Armed-Forces"
    ]
    try:
        return quantifier.index(occupation)
    except ValueError:
        return -1

def quant_relationship(relationship):
    quantifier = [
        "Wife", "Own-child", "Husband", "Not-in-family", 
        "Other-relative", "Unmarried"
    ]
    try:
        return quantifier.index(relationship)
    except ValueError:
        return -1

def quant_race(race):
    quantifier = [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", 
        "Other", "Black"
    ]
    try:
        return quantifier.index(race)
    except ValueError:
        return -1

def quant_sex(sex):
    quantifier = [
        "Female", "Male"
    ]
    try:
        return quantifier.index(sex)
    except ValueError:
        return -1

def quant_native_country(native_country):
    quantifier = [
        "United-States", "Cambodia", "England", "Puerto-Rico", 
        "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", 
        "Japan", "Greece", "South", "China", "Cuba", "Iran", 
        "Honduras", "Philippines", "Italy", "Poland", "Jamaica", 
        "Vietnam", "Mexico", "Portugal", "Ireland", "France", 
        "Dominican-Republic", "Laos", "Ecuador", "Taiwan", 
        "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", 
        "Scotland", "Thailand", "Yugoslavia", "El-Salvador", 
        "Trinidad&Tobago", "Peru", "Hong", "Holand-Netherlands"
    ]
    try:
        return quantifier.index(native_country)
    except ValueError:
        return -1

def generate_predictions(model, people):
    with open("predictions.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Prediction"])
        for person in people:
            prediction = model.predict(np.array(person[1:]).reshape(1, -1), return_probabilities=True)[0][1]
            ID = person[0]
            writer.writerow([ID, prediction])


if __name__ =='__main__':
    training_data, training_labels = read_csv("train_final.csv")
    shuffled_training_data, shuffled_training_labels = shuffle_data(training_data, training_labels)

    # Split dataset into training set and test set
    X_train, y_train, X_test, y_test = split_data(shuffled_training_data, shuffled_training_labels, 0.9)

    # Models to use
    tree = DecisionTree(max_depth=10, min_samples_leaf=1, min_information_gain=0.01)
    forest = RandomForestClassifier(n_base_learner=50, numb_of_features_splitting="sqrt")
    ada = AdaBoostClassifier(n_base_learner=10)
    svm = PrimalSVM(400, init_learning_rate=0.01, lr_func='a', max_epochs=100)

    model = forest
    model.train_with_oob(X_train, y_train)

    # Let's see the Train performance
    train_preds = model.predict(X_train)
    print("TRAIN PERFORMANCE")
    print("Train size", len(y_train))
    print("True preds", sum(train_preds == y_train))
    print("Train Accuracy", sum(train_preds == y_train) / len(y_train))
  
    # Let's see the Test performance
    test_preds = model.predict(X_test)
    print("TEST PERFORMANCE")
    print("Test size", len(y_test))
    print("True preds", sum(test_preds == y_test))
    print("Accuracy", sum(test_preds == y_test) / len(y_test))


    graded_test_data, empty = read_csv("test_final.csv")
    generate_predictions(model, graded_test_data)