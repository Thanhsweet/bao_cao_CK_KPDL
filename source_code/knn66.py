import math
from collections import Counter

TRAIN_DATA = [
    ((3.0, 4.0), "A"),
    ((1.5, 5.0), "A"),
    ((6.0, 8.0), "A"),
    ((2.5, 5.0), "A"),
    ((5.0, 2.5), "B"),
    ((5.0, 2.0), "B"),
    ((8.0, 3.0), "B"),
    ((4.5, 3.0), "B"),
]

TEST_DATA = {
    9: (8.0, 1.5),
    10: (4.5, 7.0),
}


def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def predict_knn(train_data, test_point, k):
    if not 1 <= k <= len(train_data):
        raise ValueError("k phai nam trong khoang tu 1 den so mau huan luyen")

    distances = []
    for train_point, label in train_data:
        distance = euclidean_distance(train_point, test_point)
        distances.append((distance, train_point, label))

    nearest_neighbors = sorted(distances, key=lambda item: item[0])[:k]
    votes = Counter(label for _, _, label in nearest_neighbors)
    predicted_label = votes.most_common(1)[0][0]
    return predicted_label, nearest_neighbors


def main():
    k = 3
    for sample_id, point in TEST_DATA.items():
        label, neighbors = predict_knn(TRAIN_DATA, point, k)
        print(f"Mau {sample_id} {point}: du doan lop {label}")
        print(f"  {k} lang gieng gan nhat:")
        for distance, neighbor_point, neighbor_label in neighbors:
            print(
                f"    diem={neighbor_point}, lop={neighbor_label}, "
                f"khoang cach={distance:.3f}"
            )


if __name__ == "__main__":
    main()
