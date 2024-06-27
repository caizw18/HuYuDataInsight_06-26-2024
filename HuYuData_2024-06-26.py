import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt, pi, exp

# Sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [2, 3, 4, 5, 6, 7],
    'label': [0, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
dataset = df.values

def separate_by_class(dataset):
    separated = defaultdict(list)
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        separated[class_value].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(attribute), stdev(attribute), len(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]  # remove the label
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, test_set):
    predictions = []
    for row in test_set:
        output = predict(summaries, row)
        predictions.append(output)
    return predictions

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Prepare data
training_set = dataset[:4]
test_set = dataset[4:]

# Train model
summaries = summarize_by_class(training_set)

# Test model
predictions = get_predictions(summaries, test_set)

# Evaluate predictions
actual = [row[-1] for row in test_set]
accuracy = accuracy_metric(actual, predictions)
print(f'Accuracy: {accuracy}%')