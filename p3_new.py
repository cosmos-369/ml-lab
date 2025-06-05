import pandas as pd
from collections import Counter
import math

# Load dataset
file_path = "PlayTennis.csv"  # Ensure the correct dataset is used
tennis = pd.read_csv(file_path)
print("\nGiven Play Tennis Data Set:\n", tennis)

# Entropy  Calculation
def entropy(alist):
    c = Counter(alist)
    instances = len(alist)
    prob = [x / instances for x in c.values()]
    return sum([-p * math.log2(p) for p in prob if p > 0])

# Information Gain Calculation
def information_gain(d, split, target):
    splitting = d.groupby(split)
    n = len(d.index)
    agent = splitting.agg({target: [entropy, lambda x: len(x) / n]})[target]
    agent.columns = ['Entropy', 'Proportion']
    new_entropy = sum(agent['Entropy'] * agent['Proportion'])
    return entropy(d[target]) - new_entropy

# ID3 Decision Tree Algorithm
def id3(sub, target, attributes, depth=0):
    count = Counter(sub[target])
    
    # Base case: If only one class remains, return it
    if len(count) == 1:
        return next(iter(count))
    
    # Base case: If no attributes left, return majority class
    if not attributes:
        return count.most_common(1)[0][0]
    
    # Compute information gain for each attribute
    gain = [information_gain(sub, attr, target) for attr in attributes]
    max_gain = max(gain)
    best = attributes[gain.index(max_gain)]
    
    print(f"{' ' * depth}Best Attribute: {best}, Gain: {max_gain}")
    
    tree = {best: {}}
    remaining = [i for i in attributes if i != best]
    
    # Recursively construct the decision tree
    for val, subset in sub.groupby(best):
        if subset.empty:
            tree[best][val] = count.most_common(1)[0][0]  # Assign majority class
        else:
            tree[best][val] = id3(subset, target, remaining, depth + 2)
    
    return tree

# Extract attribute names
attributes = list(tennis.columns)
attributes.remove('PlayTennis')

# Build and display the decision tree
tree = id3(tennis, 'PlayTennis', attributes)
print("\nThe Resultant Decision Tree:\n", tree)

# Classify a new sample
def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree  # Leaf node reached
    
    root = next(iter(tree))  # Get the root node
    value = sample[root]
    subtree = tree[root].get(value, None)
    
    if subtree is None:
        return "Unknown"  # If value is not in the tree, return default
    
    return classify(subtree, sample)  # Recursive call

# Example: Classify a new sample
new_sample = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
prediction = classify(tree, new_sample)
print("\nClassification Result for Sample:", new_sample, "=>", prediction)
