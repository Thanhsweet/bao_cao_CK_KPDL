training_data = [
    (("Sunny", "Hot", "High", "False"), "No"),
    (("Sunny", "Hot", "High", "True"), "No"),
    (("Overcast", "Hot", "High", "False"), "Yes"),
    (("Rainy", "Mild", "High", "False"), "Yes"),
    (("Rainy", "Cool", "Normal", "False"), "Yes"),
    (("Rainy", "Cool", "Normal", "True"), "No"),
    (("Overcast", "Cool", "Normal", "True"), "Yes"),
    (("Sunny", "Mild", "High", "False"), "No"),
    (("Sunny", "Cool", "Normal", "False"), "Yes"),
    (("Rainy", "Mild", "Normal", "False"), "Yes"),
    (("Sunny", "Mild", "Normal", "True"), "Yes"),
    (("Overcast", "Mild", "High", "True"), "Yes"),
    (("Overcast", "Hot", "Normal", "False"), "Yes"),
    (("Rainy", "Mild", "High", "True"), "No"),
]

test_sample = ("Sunny", "Cool", "High", "True")


def calculate_scores(data, sample):
    total_samples = len(data)
    class_labels = {label for _, label in data}
    scores = {}

    for label in class_labels:
        class_samples = [features for features, outcome in data if outcome == label]
        score = len(class_samples) / total_samples

        for index, value in enumerate(sample):
            matching_count = sum(
                1 for features in class_samples if features[index] == value
            )
            score *= matching_count / len(class_samples)

        scores[label] = score

    return scores


scores = calculate_scores(training_data, test_sample)
predicted_label = max(scores, key=scores.get)

print("Diem hau nghiem chua chuan hoa:")
for label, score in sorted(scores.items()):
    print(f"Play = {label}: {score:.8f}")

print(f"Nhan du doan cho mau {test_sample}: {predicted_label}")