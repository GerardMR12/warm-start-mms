from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import random
import numpy as np
from time import perf_counter
import scipy.io
from scipy.interpolate import splprep, splev


############## Functions ##############
def generate_random_matrix(
    rows_count: int, cols_count: int, p: float
) -> list[list[int]]:
    """
    Generate a random matrix with 1s (walkable) and 0s (no-step zone).
    """
    # Generate a random matrix with 1s (walkable) and 0s (no-step zone)
    visual_matrix = random_matrix(rows_count, cols_count, p)

    # Add a clear zone around the start and end
    visual_matrix = add_safe_radius(visual_matrix, 5)

    # Check and reverse matrix
    matrix = check_and_reverse_matrix(visual_matrix)
    return matrix


def check_and_reverse_matrix(matrix: list[list[int]]) -> list[list[int]]:
    """
    Check and correct the matrix.
    """
    # Assert that the start and end are walkable
    assert matrix[-1][0] == 1
    assert matrix[0][-1] == 1

    # Reverse row order for the grid
    matrix = matrix[::-1]
    return matrix


def random_matrix(rows_count: int, cols_count: int, p: float) -> list[list[int]]:
    """
    Generate a random matrix with 1s (walkable) and 0s (no-step zone).
    """
    # Create a gigantic 50x50 map matrix (1 = walkable, 0 = no-step zone)
    return [
        [1 if random.random() > p else 0 for _ in range(cols_count)]
        for _ in range(rows_count)
    ]


def draw_NFZ_radius(
    matrix: list[list[int]],
    rows_count: int,
    cols_count: int,
    min_radius: int,
    max_radius: int,
) -> list[list[int]]:
    """
    Draw a radius around the no-step zones.
    """
    # Identify no-step zones
    nfz_centers = []
    for r in range(rows_count):
        for c in range(cols_count):
            if matrix[r][c] == 0:
                nfz_centers.append((r, c))

    # Add a random radius to every no-step zone
    for r_center, c_center in nfz_centers:
        radius = random.randint(min_radius, max_radius)
        for r in range(
            max(0, r_center - radius), min(rows_count, r_center + radius + 1)
        ):
            for c in range(
                max(0, c_center - radius), min(cols_count, c_center + radius + 1)
            ):
                if (r - r_center) ** 2 + (c - c_center) ** 2 <= radius**2:
                    matrix[r][c] = 0
    return matrix


def add_safe_radius(matrix: list[list[int]], safe_radius: int) -> list[list[int]]:
    """
    Add a safe radius around the no-step zones.
    """
    for i in range(safe_radius):
        for j in range(safe_radius - i):
            matrix[-(i + 1)][j] = 1
            matrix[i][-(j + 1)] = 1
    return matrix


def generate_NFZ_matrix(rows_count: int, cols_count: int) -> list[list[int]]:
    """
    Generate a random NFZ matrix with 1s (walkable) and 0s (no-step zone).
    """
    # Create a random matrix with 1s (walkable) and 0s (no-step zone)
    visual_matrix = random_matrix(rows_count, cols_count, 0.002)

    # Draw a radius around the no-step zones
    visual_matrix = draw_NFZ_radius(visual_matrix, rows_count, cols_count, 3, 6)

    # Add a clear zone around the start and end
    visual_matrix = add_safe_radius(visual_matrix, 5)

    # Check and reverse matrix
    matrix = check_and_reverse_matrix(visual_matrix)
    return matrix


def find_path(
    matrix: list[list[int]], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Find a path using A* algorithm.
    """
    grid = Grid(matrix=matrix)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.if_at_most_one_obstacle)
    start = grid.node(start[0], start[1])
    end = grid.node(end[0], end[1])
    path, runs = finder.find_path(start, end, grid)
    return path, runs


def plot_path(
    matrix: list[list[int]],
    path: list[tuple[int, int]],
    save_path: str = "astar_path.png",
):
    """
    Plot the path on the matrix.
    """
    fig, ax = plt.subplots()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, color="grey"))
    ax.plot(
        [node.x for node in path],
        [node.y for node in path],
        "b-",
        linewidth=2,
        label="Path",
    )
    ax.plot(start[0], start[1], "go", markersize=8, label="Start")
    ax.plot(end[0], end[1], "rX", markersize=8, label="End")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("A*-solved Path")
    ax.set_xlim(0, len(matrix[0]))
    ax.set_ylim(0, len(matrix))
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")


def path_coordinates_3D(
    path: list[tuple[int, int]], initial_altitude: int, final_altitude: int
) -> list[tuple[int, int, int]]:
    """
    Convert a 2D path to a 3D path with altitude.
    """
    alt_increment = (final_altitude - initial_altitude) / (len(path) - 1)
    return [
        (node.x, node.y, initial_altitude + i * alt_increment)
        for i, node in enumerate(path)
    ]


def interpolate_path(
    path: list[tuple[float, float, float]], num_samples: int
) -> list[tuple[float, float, float]]:
    """
    Interpolate the 3D path using a spline and sample it.
    """
    if len(path) < 2:
        return path

    # Separate coordinates
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    z = [p[2] for p in path]

    # Create spline
    # k=3 (cubic) requires at least 4 points. Ensure k < len(path).
    k = min(3, len(path) - 1)

    try:
        # s=0 forces the spline to pass through all points
        tck, u = splprep([x, y, z], s=0, k=k)
    except Exception as e:
        print(f"Spline interpolation failed: {e}")
        return path

    # Sample the spline
    u_new = np.linspace(0, 1, num_samples)
    new_points = splev(u_new, tck)

    return list(zip(new_points[0], new_points[1], new_points[2]))


def increments_3D(xyz_list: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """
    Convert a 3D path to a list of increments.
    """
    # Create diff with numpy
    return np.diff(xyz_list, axis=0)


def solve_3D_non_control_case(
    dxyz_list: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float64]:
    """
    Solve the 3D non-control case. We have N points in 3D space, and N-1 segments.
    We want to find the velocity and acceleration for each segment.
    """
    v = 69  # m/s
    g = 9.81  # m/s^2
    m = 4000  # kg
    rho = 1.225  # kg/m^3
    sw = 28  # m^2
    k0 = 0.048
    k2 = 0.054187
    distance_scale_x = 40  # m
    distance_scale_y = 70  # m
    gamma_0_deg = 0  # degrees
    xi_0_deg = 45  # degrees

    # Give dimensions to the (x,y) coordinates of the dxyz_list
    dxyz_list[:, 0] *= distance_scale_x
    dxyz_list[:, 1] *= distance_scale_y

    # Create the delta_d and delta_t vectors (N-1 values)
    delta_d = np.sqrt(
        dxyz_list[:, 0] ** 2 + dxyz_list[:, 1] ** 2 + dxyz_list[:, 2] ** 2
    )
    delta_t = delta_d / v

    # Assemble all the gamma and xi values (N values)
    gamma, xi = find_gamma_and_xi(dxyz_list, delta_d, gamma_0_deg, xi_0_deg)

    # Find the increments of both gamma and xi
    delta_gamma = np.diff(gamma)
    delta_xi = np.diff(xi)

    # Solve the system of equations to find mu, CL and T
    mu, lift_coefficient, thrust = find_u(
        v, g, m, rho, sw, k0, k2, gamma, delta_gamma, delta_xi, delta_t
    )

    # Now guess the time it takes to reach the final altitude
    final_time = np.sum(delta_t)

    return mu, lift_coefficient, thrust, final_time


def find_gamma_and_xi(
    dxyz_list: np.ndarray,
    delta_d: np.ndarray,
    gamma_0_deg: float,
    xi_0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find all the gamma and xi values.
    """
    # The initial values for gamma and xi are 0
    gamma = np.zeros(len(dxyz_list) + 1)
    xi = np.zeros(len(dxyz_list) + 1)

    # Initial values for gamma and xi are 0
    gamma[0] = np.radians(gamma_0_deg)
    xi[0] = np.radians(xi_0_deg)

    # The intermediate values for gamma and xi are calculated using the arctan2 function
    for i in range(len(dxyz_list)):
        gamma[i + 1] = np.arcsin(dxyz_list[i][2] / delta_d[i])
        xi[i + 1] = np.arctan2(dxyz_list[i][1], dxyz_list[i][0])

    return gamma, xi


def find_u(
    v: float,
    g: float,
    m: float,
    rho: float,
    sw: float,
    k0: float,
    k2: float,
    gamma: np.ndarray,
    delta_gamma: np.ndarray,
    delta_xi: np.ndarray,
    delta_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the control variables mu, CL and T.
    """
    # Find mu
    mu = np.arctan2(
        -(v * np.cos(gamma[1:]) * delta_xi),
        (v * delta_gamma + g * delta_t * np.cos(gamma[1:])),
    )

    # Find lift coefficient
    lift_coefficient = (m * v * delta_gamma + m * g * delta_t * np.cos(gamma[1:])) / (
        0.5 * delta_t * rho * v**2 * sw * np.cos(mu)
    )

    # Find thrust
    drag_coefficient = k0 + k2 * lift_coefficient**2
    drag_force = 0.5 * rho * v**2 * sw * drag_coefficient
    thrust = drag_force + m * g * np.sin(gamma[1:])

    return mu, lift_coefficient, thrust


def plot_control_variables(
    mu: np.ndarray,
    lift_coefficient: np.ndarray,
    thrust: np.ndarray,
    save_path: str = "docs/astar_control_variables.png",
):
    """
    Plot the control variables.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(mu)
    ax1.set_title("Roll rotation rate over time")
    ax1.set_ylabel("mu [rad]")
    ax2.plot(lift_coefficient)
    ax2.set_title("Lift coefficient over time")
    ax2.set_ylabel("lift coefficient [-]")
    ax3.plot(thrust)
    ax3.set_title("Thrust over time")
    ax3.set_ylabel("thrust [N]")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")


############## Logic ##############
DEBUG = False
start_time = perf_counter()

# Initial and final altitude
initial_altitude = 400
final_altitude = 800
num_segments = 70

# Generate a random matrix
matrix = generate_random_matrix(100, 100, 0.0)
rows = len(matrix)
cols = len(matrix[0])

# Define start and end considering that (0, 0 is bottom-left)
start = (0, 0)  # bottom-left
end = (cols - 1, rows - 1)  # top-right

# Find the path
path, runs = find_path(matrix, start, end)

# Print information about the path
print("Operations:", runs, "Path length:", len(path))
if len(path) == 0:
    print("No path found")
print("Path:", " -> ".join(f"({node.x}, {node.y})" for node in path)) if DEBUG else None

print(f"Path finding time allocation: {perf_counter() - start_time:.2f} seconds")

# Get (x, y, z) coordinates
xyz_coordinates = path_coordinates_3D(path, initial_altitude, final_altitude)
print("XYZ coordinates:", xyz_coordinates) if DEBUG else None

# Create a spline out of the coordinate and then sample the spline for X points
xyz_coordinates = interpolate_path(xyz_coordinates, num_segments + 1)
print("Interpolated coordinates:", xyz_coordinates) if DEBUG else None

# Get (x, y, z) coordinates derivative
dxyz_coordinates = increments_3D(xyz_coordinates)
print("Coordinates derivative:", dxyz_coordinates) if DEBUG else None

# Problem definition
print(f"We have {len(xyz_coordinates)} points and {len(dxyz_coordinates)} segments.")

# Find the control variables for the 3D non-control case
mu, lift_coefficient, thrust, final_time = solve_3D_non_control_case(dxyz_coordinates)

print(f"Total time allocation: {perf_counter() - start_time:.2f} seconds")

# Print information about the control variables
print("Values of mu:", mu) if DEBUG else None
print("Values of lift coefficient:", lift_coefficient) if DEBUG else None
print("Values of thrust:", thrust) if DEBUG else None
assert (
    len(mu) == len(lift_coefficient) == len(thrust)
), "The length of mu, lift coefficient, and thrust must be the same."
print("Length of mu, lift coefficient, thrust:", len(mu))
print(f"Trajectory time: {final_time:.2f} seconds")

# Create a directory for the results
results_dir = "docs"
os.makedirs(results_dir, exist_ok=True)

# Plot the path
plot_path(matrix, path, f"{results_dir}/astar_path.png")

# Plot the control variables
plot_control_variables(
    mu, lift_coefficient, thrust, f"{results_dir}/astar_control_variables.png"
)

# Save a .mat file with the control variables
guess_results = [final_time.item()]
guess_results.extend(thrust.tolist())
guess_results.extend(lift_coefficient.tolist())
guess_results.extend(mu.tolist())
scipy.io.savemat(
    f"{results_dir}/astar_guess_results.mat", {"python_initial_guesses": guess_results}
)
print(f"Guess results saved to {results_dir}/astar_guess_results.mat")
