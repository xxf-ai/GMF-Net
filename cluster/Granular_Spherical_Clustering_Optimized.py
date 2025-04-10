import gc
import os
import psutil
import time
from math import exp, sqrt
import numpy as np
from sklearn import metrics
from sklearn.cluster import SpectralClustering, KMeans
import traceback

try:
    from Granular_Spherical_Clustering import generate_granular_balls
except ImportError:
    print("ERROR: Ensure 'Granular_Spherical_Clustering.py' is in the same directory or accessible.")
    exit()


class BKOAOptimizer:
    def __init__(self, objective_function, bounds, pop_size=20, max_iter=50, args=(), timeout=300):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.args = args
        self.dimension = len(bounds)
        self.timeout = timeout

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.alpha = 0.8
        self.beta = 0.5
        self.gamma = 0.1

        self.early_stop_threshold = 1e-6
        self.early_stop_count = 0
        self.early_stop_max = 5

    def _ensure_bounds(self, position):
        position = np.clip(position, self.bounds[:, 0], self.bounds[:, 1])
        position[0] = max(2, int(round(position[0])))
        position[1] = max(1e-6, position[1])
        return position

    def run(self):
        population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.pop_size, self.dimension))
        fitness = np.full(self.pop_size, float('inf'))

        # Initialize a small subset first
        init_pop_size = min(self.pop_size, 10)
        print(f"  Initializing BKOA population ({init_pop_size}/{self.pop_size})...")
        for i in range(init_pop_size):
            population[i] = self._ensure_bounds(population[i])
            fitness[i] = self.objective_function(population[i], *self.args)
            if fitness[i] < self.global_best_fitness:
                self.global_best_fitness = fitness[i]
                self.global_best_position = population[i].copy()
            print(f"  Init progress: {i + 1}/{init_pop_size}, Current Fitness: {fitness[i]:.4f}   ", end='\r')
        print()

        # Initialize rest of population, possibly guided by the initial best
        if self.global_best_position is not None:
            print(f"  Guiding remaining population initialization using initial best.")
            bound_range = self.bounds[:, 1] - self.bounds[:, 0]
            for i in range(init_pop_size, self.pop_size):
                noise = np.random.normal(0, 0.1 * bound_range, self.dimension)
                population[i] = self._ensure_bounds(self.global_best_position + noise)
                fitness[i] = float('inf')
        else:
            print(f"  Randomly initializing remaining population.")
            for i in range(init_pop_size, self.pop_size):
                population[i] = self._ensure_bounds(
                    np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dimension)))
                fitness[i] = float('inf')

        start_time = time.time()
        previous_best_fitness = self.global_best_fitness

        for t in range(self.max_iter):
            print(f"  BKOA Iteration: {t + 1}/{self.max_iter}, Best Fitness (-ARI): {self.global_best_fitness:.6f}    ",
                  end='\r')

            if time.time() - start_time > self.timeout:
                print(f"\n  BKOA optimization timed out ({self.timeout}s), terminating early.")
                break

            fitness_change = abs(previous_best_fitness - self.global_best_fitness)
            if fitness_change < self.early_stop_threshold:
                self.early_stop_count += 1
                if self.early_stop_count >= self.early_stop_max:
                    print(
                        f"\n  BKOA converged, terminating early (Threshold={self.early_stop_threshold}, Count={self.early_stop_count}).")
                    break
            else:
                self.early_stop_count = 0
            previous_best_fitness = self.global_best_fitness

            # Evaluate fitness for individuals marked as inf
            for i in range(self.pop_size):
                if fitness[i] == float('inf'):
                    population[i] = self._ensure_bounds(population[i])
                    fitness[i] = self.objective_function(population[i], *self.args)

            # Sort population by fitness (ascending, since we minimize -ARI)
            sort_indices = np.argsort(fitness)
            population = population[sort_indices]
            fitness = fitness[sort_indices]

            # Update global best if current best is better
            if fitness[0] < self.global_best_fitness:
                self.global_best_fitness = fitness[0]
                self.global_best_position = population[0].copy()

            current_iter_best_pos = population[0]
            active_size = max(3, self.pop_size // 2)

            # Update active individuals (excluding the best one)
            for i in range(1, active_size):
                # Select another random individual from the active set
                other_active_indices = np.delete(np.arange(active_size), i)
                if not len(other_active_indices): continue  # Should not happen if active_size >= 2
                rand_idx = np.random.choice(other_active_indices)
                rand_pos = population[rand_idx]

                # Simplified BKOA movement
                r1, r2 = np.random.rand(2)
                term1 = self.alpha * r1 * (current_iter_best_pos - population[i])
                term2 = self.beta * r2 * (rand_pos - population[i])
                term3 = self.gamma * (np.random.rand(self.dimension) - 0.5) * (self.bounds[:, 1] - self.bounds[:, 0])

                new_position = population[i] + term1 + term2 + term3
                new_position = self._ensure_bounds(new_position)
                new_fitness = self.objective_function(new_position, *self.args)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

            # Update non-active individuals (exploration around current best)
            bound_range = self.bounds[:, 1] - self.bounds[:, 0]
            for i in range(active_size, self.pop_size):
                noise = np.random.normal(0, 0.2 * bound_range, self.dimension)
                new_position = self._ensure_bounds(current_iter_best_pos + noise)
                population[i] = new_position
                fitness[i] = float('inf')

                # Final check for best position if it was never updated
        if self.global_best_position is None:
            print("\nWarning: Global best position was not updated. Selecting best from final population.")
            if len(population) > 0:
                # Find the minimum fitness that is not infinity
                valid_fitness_indices = np.where(fitness != float('inf'))[0]
                if len(valid_fitness_indices) > 0:
                    final_best_idx = valid_fitness_indices[np.argmin(fitness[valid_fitness_indices])]
                    self.global_best_position = population[final_best_idx].copy()
                    self.global_best_fitness = fitness[final_best_idx]
                else:
                    print("ERROR: All final fitness values are infinity. Cannot determine best.")
                    # Return a default or raise error
                    default_pos = self._ensure_bounds(self.bounds.mean(axis=1))
                    return default_pos, float('inf')
            else:
                print("ERROR: Population is empty. Cannot determine best.")
                default_pos = self._ensure_bounds(self.bounds.mean(axis=1))
                return default_pos, float('inf')

        print(f"\nBKOA Optimization Finished. Final Best Fitness (-ARI): {self.global_best_fitness:.6f}")
        final_best_position = self._ensure_bounds(self.global_best_position)
        return final_best_position, self.global_best_fitness


class GranularBallRepresentation:
    def __init__(self, points, label):
        self.points = points
        self.center = self.points.mean(0) if len(points) > 0 else np.array([])
        self.label = label
        self.radius = self._calculate_radius()

    def _calculate_radius(self):
        if self.points.shape[0] <= 1 or self.center.size == 0:
            return 0.0
        distances = np.linalg.norm(self.points - self.center, axis=1)
        return np.max(distances) if len(distances) > 0 else 0.0


def calculate_affinity(center1, center2, radius1, radius2, delta_squared_term):
    if center1.size == 0 or center2.size == 0: return 0.0
    distance = np.linalg.norm(center1 - center2)
    gap = distance - radius1 - radius2

    # Handle zero or near-zero delta
    if delta_squared_term <= 1e-12:
        return 1.0 if gap < 1e-9 else 0.0

        # Exponential kernel based on the gap
    affinity = exp(-gap / delta_squared_term) if gap > 0 else 1.0
    return max(0, affinity)  # Ensure non-negative


def create_ball_dictionary(ball_data_list):
    ball_dict = {}
    valid_ball_count = 0
    for i, points in enumerate(ball_data_list):
        if len(points) > 0:
            gb_repr = GranularBallRepresentation(points, valid_ball_count)
            # Check if ball is valid (has a center) before adding
            if gb_repr.center.size > 0:
                ball_dict[valid_ball_count] = gb_repr
                valid_ball_count += 1
            else:
                print(f"Warning: Skipping ball {i} with {len(points)} points as it has no center.")
    return ball_dict


def perform_clustering_and_evaluate(ball_dict, num_clusters, delta_param, original_features, original_ground_truth):
    clustering_method_used = "Spectral"
    try:
        ball_keys = list(ball_dict.keys())
        num_balls = len(ball_keys)

        if num_balls == 0:
            print("ERROR: No valid granular balls provided for clustering.")
            return 0.0, np.full(len(original_features), -1, dtype=int)
        if num_clusters <= 0:
            print(f"ERROR: Invalid number of clusters requested: {num_clusters}")
            return 0.0, np.full(len(original_features), -1, dtype=int)

        # Ensure num_clusters is not greater than the number of balls
        if num_clusters > num_balls:
            num_clusters = num_balls
        if num_clusters == 1:
            pass

        # Extract centers and radii only from valid balls present in the dict
        ball_centers = np.array([ball_dict[key].center for key in ball_keys])
        ball_radii = np.array([ball_dict[key].radius for key in ball_keys])
        valid_ball_keys = ball_keys  # All keys in dict are assumed valid now
        num_valid_balls = len(valid_ball_keys)

        if num_valid_balls < num_clusters:
            num_clusters = num_valid_balls
            if num_clusters <= 0:
                print("ERROR: No valid balls left after filtering.")
                return 0.0, np.full(len(original_features), -1, dtype=int)

        # Calculate affinity matrix
        affinity_matrix = np.zeros((num_valid_balls, num_valid_balls))
        delta_squared_term = 2 * (delta_param ** 2) if delta_param > 1e-9 else 1e-12

        for i in range(num_valid_balls):
            affinity_matrix[i, i] = 1.0
            for j in range(i + 1, num_valid_balls):
                affinity = calculate_affinity(
                    ball_centers[i], ball_centers[j],
                    ball_radii[i], ball_radii[j],
                    delta_squared_term
                )
                affinity_matrix[i, j] = affinity
                affinity_matrix[j, i] = affinity

        # Perform clustering
        if num_clusters == 1:
            ball_cluster_labels = np.zeros(num_valid_balls, dtype=int)
        else:
            try:
                spectral = SpectralClustering(
                    n_clusters=num_clusters,
                    affinity="precomputed",
                    assign_labels="discretize",
                    random_state=42,
                    n_init=10,
                    n_jobs=-1
                )

                affinity_matrix = np.nan_to_num(affinity_matrix)
                affinity_matrix = np.maximum(affinity_matrix, 0)

                affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

                ball_cluster_labels = spectral.fit_predict(affinity_matrix)
                clustering_method_used = "Spectral"
            except Exception as spectral_error:
                print(
                    f"\nSpectral Clustering failed (n_clusters={num_clusters}, delta={delta_param:.4f}): {spectral_error}. Trying K-Means fallback...")
                try:
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    ball_cluster_labels = kmeans.fit_predict(ball_centers)
                    clustering_method_used = "KMeans_Fallback"
                except Exception as kmeans_error:
                    print(f"K-Means fallback also failed: {kmeans_error}")
                    traceback.print_exc()
                    return 0.0, np.full(len(original_features), -1, dtype=int)

        final_point_labels = np.full(len(original_features), -1, dtype=int)

        valid_key_to_cluster_map = {valid_ball_keys[i]: ball_cluster_labels[i] for i in range(num_valid_balls)}

        if num_valid_balls > 0:
            for point_idx, point in enumerate(original_features):
                distances_sq = np.sum((ball_centers - point) ** 2, axis=1)
                nearest_ball_idx = np.argmin(distances_sq)
                nearest_ball_key = valid_ball_keys[nearest_ball_idx]

                if nearest_ball_key in valid_key_to_cluster_map:
                    final_point_labels[point_idx] = valid_key_to_cluster_map[nearest_ball_key]

        # Calculate Adjusted Rand Index (ARI)
        valid_indices = np.where((original_ground_truth != -1) & (final_point_labels != -1))[0]
        if len(valid_indices) < 2:
            ari_score = 0.0
        else:
            try:
                ari_score = metrics.adjusted_rand_score(
                    original_ground_truth[valid_indices],
                    final_point_labels[valid_indices]
                )
            except Exception as ari_error:
                print(f"Error calculating ARI: {ari_error}")
                ari_score = 0.0

        return ari_score, final_point_labels

    except Exception as e:
        print(f"\nSevere error during clustering process: {e}")
        traceback.print_exc()
        return 0.0, np.full(len(original_features), -1, dtype=int)


def bkoa_objective(params, ball_dict, original_features, original_ground_truth):
    """ BKA objective function: minimize negative ARI """
    num_clusters = int(params[0])
    continuous_delta = float(params[1])

    # Discretize delta to one decimal place within [0.1, 1.0]
    snapped_delta = round(continuous_delta * 10.0) / 10.0
    min_discrete_delta = 0.1
    max_discrete_delta = 1.0
    final_discrete_delta = max(min_discrete_delta, min(max_discrete_delta, snapped_delta))

    if num_clusters < 1:
        return 1.0

    ari_score, _ = perform_clustering_and_evaluate(
        ball_dict, num_clusters, final_discrete_delta, original_features, original_ground_truth)

    fitness = -ari_score
    return fitness


def main():
    original_datasets = {'test': 3}

    datasets_to_run = ['test']

    # BKA Configuration
    BKOA_POP_SIZE = 3
    BKOA_MAX_ITER = 2
    BKOA_TIMEOUT_SECONDS = 30

    optimization_results = {}

    DATASET_DIRECTORY = "D:\GSC\dataset"

    if not os.path.isdir(DATASET_DIRECTORY):
        print(f"ERROR: Dataset directory not found: '{DATASET_DIRECTORY}'")
        print("Please modify the DATASET_DIRECTORY variable in the script.")
        return

    for dataset_name in datasets_to_run:
        print(f"Processing Dataset: {dataset_name}")
        print("-" * 40)

        n_clusters_hint = original_datasets.get(dataset_name, 3)
        delta_hint = 0.1

        continuous_delta_lower = 0.05
        continuous_delta_upper = 1.05

        n_clusters_lower = max(2, int(n_clusters_hint * 0.7))
        n_clusters_upper = max(n_clusters_lower + 2, int(n_clusters_hint * 1.5) + 1)

        parameter_bounds = [
            [n_clusters_lower, n_clusters_upper],
            [continuous_delta_lower, continuous_delta_upper]
        ]
        print(
            f"Parameter Bounds: n_clusters={parameter_bounds[0]}, continuous_delta=[{parameter_bounds[1][0]:.3f}, {parameter_bounds[1][1]:.3f}]")
        print(f"  (delta will be discretized to [0.1, 1.0] step 0.1 for evaluation)")

        gc.collect()
        time_total_start = time.time()

        # === Step 1: Generate Granular Balls ===
        print("Step 1: Generating Granular Balls...")
        time_gb_start = time.time()
        try:
            # Construct full path for the dataset file
            dataset_file_path_prefix = os.path.join(DATASET_DIRECTORY, "data_")
            ground_truth, features, ball_data_list, num_data_points = generate_granular_balls(
                dataset_name,
                data_path_prefix=dataset_file_path_prefix
            )
        except FileNotFoundError:
            # Error message printed within generate_granular_balls
            print(f"Skipping dataset {dataset_name} due to file error.")
            continue
        except Exception as e:
            print(f"Error during granular ball generation for {dataset_name}: {e}")
            traceback.print_exc()
            continue
        time_gb_end = time.time()
        print(f"  Granular ball generation time: {time_gb_end - time_gb_start:.2f} seconds")
        print(f"  Data points: {num_data_points}, Generated raw balls: {len(ball_data_list)}")

        # === Step 2: Prepare Ball Dictionary ===
        print("Step 2: Preparing Ball Dictionary...")
        ball_dict = create_ball_dictionary(ball_data_list)
        num_valid_balls = len(ball_dict)
        if num_valid_balls == 0:
            print("ERROR: No valid granular balls were created. Skipping optimization.")
            continue
        print(f"  Number of valid balls for clustering: {num_valid_balls}")

        # Adjust cluster upper bound if necessary
        original_upper_bound = parameter_bounds[0][1]
        parameter_bounds[0][1] = min(parameter_bounds[0][1], num_valid_balls)
        # Ensure lower bound is not greater than adjusted upper bound
        parameter_bounds[0][0] = min(parameter_bounds[0][0], parameter_bounds[0][1])
        if parameter_bounds[0][1] < original_upper_bound:
            print(f"  Adjusted n_clusters upper bound to {parameter_bounds[0][1]} (number of valid balls)")
        if parameter_bounds[0][0] > parameter_bounds[0][1]:
            print(
                f"ERROR: Cluster lower bound ({parameter_bounds[0][0]}) > upper bound ({parameter_bounds[0][1]}) after adjustment.")
            continue

        # === Step 3: Optimize Parameters using BKOA ===
        print(
            f"Step 3: Optimizing Parameters with BKOA (Pop={BKOA_POP_SIZE}, Iter={BKOA_MAX_ITER}, Timeout={BKOA_TIMEOUT_SECONDS}s)...")
        time_opt_start = time.time()

        optimizer = BKOAOptimizer(
            objective_function=bkoa_objective,
            bounds=parameter_bounds,
            pop_size=BKOA_POP_SIZE,
            max_iter=BKOA_MAX_ITER,
            args=(ball_dict, features, ground_truth),  # Pass necessary args to objective
            timeout=BKOA_TIMEOUT_SECONDS
        )

        best_params_continuous, best_fitness_value = optimizer.run()

        time_opt_end = time.time()
        print(f"  BKOA optimization time: {time_opt_end - time_opt_start:.2f} seconds")

        # Extract and finalize optimized parameters
        optimized_n_clusters = int(best_params_continuous[0])
        optimized_continuous_delta = float(best_params_continuous[1])

        # Final discretization of the best delta found by BKA
        snapped_best_delta = round(optimized_continuous_delta * 10.0) / 10.0
        final_optimized_delta = max(0.1, min(1.0, snapped_best_delta))

        optimized_ari_score = -best_fitness_value  # Fitness was -ARI

        print("\n--- Optimization Results ---")
        print(f"Optimized n_clusters: {optimized_n_clusters}")
        print(
            f"Optimized discrete delta: {final_optimized_delta:.1f} (from continuous: {optimized_continuous_delta:.4f})")
        print(f"Best ARI found during optimization: {optimized_ari_score:.6f}")

        # === Step 4: Final Evaluation ===
        print("\nStep 4: Final Evaluation using optimized discrete parameters...")
        final_ari, final_cluster_labels = perform_clustering_and_evaluate(
            ball_dict, optimized_n_clusters, final_optimized_delta, features, ground_truth)

        # Calculate NMI for the final clustering
        valid_indices_final = np.where((ground_truth != -1) & (final_cluster_labels != -1))[0]
        if len(valid_indices_final) < 2:
            final_nmi = 0.0
        else:
            try:
                # Specify average_method to avoid future warnings, 'arithmetic' is common
                final_nmi = metrics.normalized_mutual_info_score(
                    ground_truth[valid_indices_final],
                    final_cluster_labels[valid_indices_final],
                    average_method='arithmetic'
                )
            except Exception as nmi_error:
                print(f"Error calculating NMI: {nmi_error}")
                final_nmi = 0.0

        print(f"Final ARI (using optimized discrete params): {final_ari:.6f}")
        print(f"Final NMI (using optimized discrete params): {final_nmi:.6f}")

        time_total_end = time.time()
        total_duration = time_total_end - time_total_start
        print(f"\nTotal time for dataset {dataset_name}: {total_duration:.2f} seconds")
        print("-" * 40 + "\n")

        # Store results
        optimization_results[dataset_name] = {
            'optimized_n_clusters': optimized_n_clusters,
            'optimized_delta': final_optimized_delta,
            'best_optimization_ari': optimized_ari_score,
            'final_ari': final_ari,
            'final_nmi': final_nmi,
            'total_time_s': total_duration,
            'num_valid_balls': num_valid_balls
        }
        # Optional: Clear memory intensive objects if running many datasets
        del ball_dict, features, ground_truth, ball_data_list, optimizer
        gc.collect()

    # === Final Summary ===
    print("\n=== Final Optimization Results Summary ===")
    if not optimization_results:
        print("No datasets were successfully processed.")
    else:
        for name, result in optimization_results.items():
            print(f"Dataset: {name} (#Valid Balls: {result['num_valid_balls']})")
            print(
                f"  Optimized Params: n_clusters={result['optimized_n_clusters']}, delta={result['optimized_delta']:.1f}")
            print(f"  Best ARI (Optimization): {result['best_optimization_ari']:.4f}")
            print(f"  Final ARI (Evaluation):  {result['final_ari']:.4f}")
            print(f"  Final NMI (Evaluation):  {result['final_nmi']:.4f}")
            print(f"  Total Time: {result['total_time_s']:.2f}s")
            print("-" * 25)


if __name__ == '__main__':
    # Basic system info printout
    print(f"CPU cores: {psutil.cpu_count(logical=True)}")
    mem_info = psutil.virtual_memory()
    print(f"Total Memory: {mem_info.total / (1024 ** 3):.2f} GB")
    main()
