import math


DATA = {
    "A1": (2, 10),
    "A2": (2, 5),
    "A3": (8, 4),
    "A4": (5, 8),
    "A5": (7, 5),
    "A6": (6, 4),
    "A7": (1, 2),
    "A8": (4, 9),
}

INITIAL_CENTROIDS = {
    "C1": DATA["A1"],
    "C2": DATA["A2"],
    "C3": DATA["A3"],
}


def euclidean_distance(point, centroid):
    return math.sqrt(
        sum((coordinate - center) ** 2 for coordinate, center in zip(point, centroid))
    )


def assign_clusters(data, centroids):
    clusters = {name: [] for name in centroids}
    assignments = {}

    for point_name, point in data.items():
        distances = {
            name: euclidean_distance(point, centroid)
            for name, centroid in centroids.items()
        }
        closest_centroid = min(distances, key=distances.get)
        assignments[point_name] = closest_centroid
        clusters[closest_centroid].append(point_name)

        distance_text = ", ".join(
            f"{name}={distance:.3f}" for name, distance in distances.items()
        )
        print(
            f"{point_name}{point}: {distance_text} "
            f"-> {closest_centroid}"
        )

    return clusters, assignments


def update_centroids(data, clusters):
    centroids = {}
    for name, point_names in clusters.items():
        if not point_names:
            raise ValueError(f"{name} has no points; cannot update its centroid")

        points = [data[point_name] for point_name in point_names]
        dimensions = len(points[0])
        centroids[name] = tuple(
            sum(point[index] for point in points) / len(points)
            for index in range(dimensions)
        )

    return centroids


def centroids_are_equal(first, second):
    return all(
        math.isclose(first[name][index], second[name][index], abs_tol=1e-12)
        for name in first
        for index in range(len(first[name]))
    )


def run_kmeans(data, initial_centroids, max_iterations=100):
    centroids = initial_centroids.copy()
    previous_assignments = None

    for iteration in range(1, max_iterations + 1):
        print(f"\nLan lap {iteration}")
        clusters, assignments = assign_clusters(data, centroids)
        updated_centroids = update_centroids(data, clusters)

        print("Tam moi:")
        for name, centroid in updated_centroids.items():
            print(f"  {name} = ({centroid[0]:.2f}, {centroid[1]:.2f})")

        if assignments == previous_assignments or centroids_are_equal(
            centroids, updated_centroids
        ):
            centroids = updated_centroids
            break

        previous_assignments = assignments
        centroids = updated_centroids
    else:
        raise RuntimeError("K-means did not converge within the iteration limit")

    print("\nKet qua cuoi cung:")
    for name, point_names in clusters.items():
        centroid = centroids[name]
        members = ", ".join(point_names)
        print(
            f"  {name}: {{{members}}}, "
            f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f})"
        )


if __name__ == "__main__":
    run_kmeans(DATA, INITIAL_CENTROIDS)
