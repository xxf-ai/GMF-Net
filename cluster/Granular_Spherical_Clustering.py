from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy.linalg as la

class GranularBall:
    def __init__(self, points, label):
        self.points = points
        self.center = self.points.mean(0) if len(points) > 0 else np.array([])
        self.label = label
        self.radius = self._calculate_radius()
        self.point_count = len(points)

    def _calculate_radius(self):
        if self.point_count == 0:
            return 0.0
        if self.point_count == 1:
            return 0.0
        distances = la.norm(self.points - self.center, axis=1)
        return np.max(distances)

def calculate_radius(points):
    num_points = len(points)
    if num_points <= 1:
        return 0.0
    center = points.mean(0)
    distances = la.norm(points - center, axis=1)
    radius = np.max(distances)
    return radius

def split_ball_by_distance(points):
    ball1_points = []
    ball2_points = []
    num_points, num_features = points.shape
    if num_points < 2:
         return [points, np.array([])]

    transposed_points = points.T
    gram_matrix = np.dot(transposed_points.T, transposed_points)
    diag_gram = np.diag(gram_matrix)
    h_matrix = np.tile(diag_gram, (num_points, 1))

    distance_matrix_sq = np.maximum(0, h_matrix + h_matrix.T - gram_matrix * 2)
    distance_matrix = np.sqrt(distance_matrix_sq)


    if np.max(distance_matrix) == 0:

         mid_idx = num_points // 2
         ball1_points = points[:mid_idx]
         ball2_points = points[mid_idx:]
         return [np.array(ball1_points), np.array(ball2_points)]

    row_indices, col_indices = np.where(distance_matrix == np.max(distance_matrix))

    valid_pair_found = False
    for r_idx, c_idx in zip(row_indices, col_indices):
        if r_idx != c_idx:
           point1_idx = r_idx
           point2_idx = c_idx
           valid_pair_found = True
           break

    if not valid_pair_found:

        mid_idx = num_points // 2
        ball1_points = points[:mid_idx]
        ball2_points = points[mid_idx:]
        return [np.array(ball1_points), np.array(ball2_points)]


    for j in range(num_points):
        dist_to_p1 = distance_matrix[j, point1_idx]
        dist_to_p2 = distance_matrix[j, point2_idx]
        if dist_to_p1 <= dist_to_p2:
            ball1_points.append(points[j, :])
        else:
            ball2_points.append(points[j, :])

    # Ensure both resulting balls are not empty if possible
    if not ball1_points:

        ball1_points.append(points[point2_idx])
        ball2_points = [p for i, p in enumerate(ball2_points) if not np.array_equal(p, points[point2_idx])]
    elif not ball2_points:
        # Move one point from ball1 to ball2
        ball2_points.append(points[point1_idx])
        ball1_points = [p for i, p in enumerate(ball1_points) if not np.array_equal(p, points[point1_idx])]


    return [np.array(ball1_points), np.array(ball2_points)]


def calculate_density(points):
    num_points = len(points)
    if num_points <= 1:
        return float(num_points)

    center = points.mean(0)
    distances = la.norm(points - center, axis=1)
    sum_radius = np.sum(distances)
    mean_radius = sum_radius / num_points if num_points > 0 else 0

    if sum_radius > 1e-9:
        density_volume = num_points / sum_radius
    else:
        density_volume = float('inf') if num_points > 0 else 0.0 # Infinite density if radius is zero

    return density_volume

def split_based_on_density(ball_list):
    new_ball_list = []
    min_points_for_split = 8
    min_points_after_split = 4

    for ball_points in ball_list:
        if len(ball_points) >= min_points_for_split:
            points_child1, points_child2 = split_ball_by_distance(ball_points)

            if len(points_child1) == 0 or len(points_child2) == 0:
                new_ball_list.append(ball_points)
                continue

            density_parent = calculate_density(ball_points)
            density_child1 = calculate_density(points_child1)
            density_child2 = calculate_density(points_child2)

            total_points = len(points_child1) + len(points_child2)
            weight1 = len(points_child1) / total_points if total_points > 0 else 0
            weight2 = len(points_child2) / total_points if total_points > 0 else 0

            weighted_child_density = (weight1 * density_child1 + weight2 * density_child2)

            density_improves = weighted_child_density > density_parent
            sufficient_points = (len(points_child1) >= min_points_after_split and
                                 len(points_child2) >= min_points_after_split)

            if density_improves and sufficient_points:
                new_ball_list.extend([points_child1, points_child2])
            else:
                new_ball_list.append(ball_points)
        else:
            new_ball_list.append(ball_points)
    return new_ball_list

def normalize_balls_by_radius(ball_list, detection_radius):
    temp_ball_list = []
    min_points_for_normalize = 2
    radius_threshold_factor = 2.0

    for ball_points in ball_list:
        if len(ball_points) < min_points_for_normalize:
            temp_ball_list.append(ball_points)
        else:
            current_radius = calculate_radius(ball_points)
            if current_radius <= radius_threshold_factor * detection_radius:
                temp_ball_list.append(ball_points)
            else:
                points_child1, points_child2 = split_ball_by_distance(ball_points)
                if len(points_child1) > 0:
                    temp_ball_list.append(points_child1)
                if len(points_child2) > 0:
                    temp_ball_list.append(points_child2)

    return temp_ball_list

def generate_granular_balls(dataset_name, data_path_prefix=""):  # Use your path："D:\cluster\datasets\data_"
    try:
        mat_data = loadmat(f"{data_path_prefix}{dataset_name}.mat")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path_prefix}{dataset_name}.mat")
        raise
    except Exception as e:
        print(f"Error loading MAT file: {e}")
        raise

    features = mat_data['fea']
    ground_truth = mat_data['gt'].flatten()

    num_points = features.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    current_ball_list = [pca_features]

    iteration_count = 0
    max_iterations = 50

    # Density-based splitting
    while iteration_count < max_iterations:
        iteration_count += 1
        ball_count_before_split = len(current_ball_list)
        current_ball_list = split_based_on_density(current_ball_list)
        ball_count_after_split = len(current_ball_list)
        if ball_count_after_split == ball_count_before_split:
            break
    if iteration_count == max_iterations:
        print("Warning: Max iterations reached during density splitting.")


    radii = [calculate_radius(ball_points) for ball_points in current_ball_list if len(ball_points) >= 2]
    if not radii:
         print("Warning: No balls with sufficient points to calculate radii.")
         detection_radius = 0.0
    else:
        radius_median = np.median(radii)
        radius_mean = np.mean(radii)
        detection_radius = max(radius_median, radius_mean, 1e-6)

    iteration_count = 0
    while iteration_count < max_iterations:
         iteration_count += 1
         ball_count_before_norm = len(current_ball_list)
         current_ball_list = normalize_balls_by_radius(current_ball_list, detection_radius)
         ball_count_after_norm = len(current_ball_list)

         if ball_count_after_norm == ball_count_before_norm:

             break
    if iteration_count == max_iterations:
        print("Warning: Max iterations reached during radius normalization.")

    final_ball_list = [ball for ball in current_ball_list if len(ball) > 0]

    return ground_truth, pca_features, final_ball_list, num_points
