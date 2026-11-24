from collections import Counter
from math import log2


training_data = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"},
]

target = "PlayTennis"
attributes = ["Outlook", "Temperature", "Humidity", "Wind"]


def entropy(data):
    label_counts = Counter(row[target] for row in data)
    total = len(data)
    return -sum(
        (count / total) * log2(count / total)
        for count in label_counts.values()
    )


def information_gain(data, attribute):
    total_entropy = entropy(data)
    values = dict.fromkeys(row[attribute] for row in data)
    remainder = 0

    for value in values:
        subset = [row for row in data if row[attribute] == value]
        remainder += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - remainder


def majority_label(data):
    return Counter(row[target] for row in data).most_common(1)[0][0]


def build_tree(data, available_attributes):
    labels = {row[target] for row in data}
    if len(labels) == 1:
        return next(iter(labels))
    if not available_attributes:
        return majority_label(data)

    best_attribute = max(
        available_attributes,
        key=lambda attribute: information_gain(data, attribute),
    )
    remaining_attributes = [
        attribute for attribute in available_attributes
        if attribute != best_attribute
    ]
    branches = {}

    for value in dict.fromkeys(row[best_attribute] for row in data):
        subset = [row for row in data if row[best_attribute] == value]
        branches[value] = build_tree(subset, remaining_attributes)

    return {
        "attribute": best_attribute,
        "default": majority_label(data),
        "branches": branches,
    }


def predict(tree, sample):
    while isinstance(tree, dict):
        attribute = tree["attribute"]
        tree = tree["branches"].get(sample.get(attribute), tree["default"])
    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}{tree}")
        return

    print(f"{indent}{tree['attribute']}")
    for value, branch in tree["branches"].items():
        if isinstance(branch, dict):
            print(f"{indent}  {value}:")
            print_tree(branch, indent + "    ")
        else:
            print(f"{indent}  {value}: {branch}")


print(f"Entropy(S) = {entropy(training_data):.3f}")
print("Information Gain:")
for attribute in attributes:
    print(f"  {attribute}: {information_gain(training_data, attribute):.3f}")

decision_tree = build_tree(training_data, attributes)
print("\nCay quyet dinh ID3:")
print_tree(decision_tree)

correct = sum(
    predict(decision_tree, row) == row[target]
    for row in training_data
)
print(f"\nDo chinh xac tren tap huan luyen: {correct}/{len(training_data)}")