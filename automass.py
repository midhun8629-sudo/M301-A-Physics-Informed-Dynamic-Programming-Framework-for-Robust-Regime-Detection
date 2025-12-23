# ==============================================================================
# PROJECT: AutoMass - Automated Effective Mass Extraction
# AUTHOR: [Your Name]
# PURPOSE: Uses Dynamic Segmentation to find the parabolic limit in DFT data.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# PART 1: PHYSICS SIMULATION (Generating Synthetic VASP Data)
# ------------------------------------------------------------------------------
def generate_band_structure(n_points=150, true_mass=0.4, noise_level=0.005):
    """
    Generates a synthetic E-k diagram (Band Structure).
    Physics: E(k) = (hbar^2 * k^2) / 2m* + Anharmonic Terms
    """
    # k-path (Momentum) usually goes from 0 (Gamma point) outwards
    k = np.linspace(0, 0.8, n_points)

    # 1. The Parabolic Term (What we want to capture)
    # Simplified units: E = A * k^2, where A = 1 / (2 * mass)
    A_true = 1.0 / (2.0 * true_mass)
    E_parabolic = A_true * (k**2)

    # 2. The Anharmonic Term (The deviation at high k)
    # Real bands flatten out or curve differently far from k=0
    E_anharmonic = -0.5 * (k**4) + 0.2 * (k**6)

    # 3. Combine and add Noise (Simulating numerical error in DFT)
    E_total = E_parabolic + E_anharmonic + np.random.normal(0, noise_level, n_points)

    return k, E_total

# Generate the data
print("--- 1. Generating Synthetic DFT Data ---")
k_axis, E_data = generate_band_structure()


# ------------------------------------------------------------------------------
# PART 2: THE ALGORITHM (Dynamic Segmented Least Squares)
# ------------------------------------------------------------------------------
def compute_quadratic_sse(x_subset, y_subset):
    """
    LEAST SQUARES METHOD: Fits y = ax^2 + bx + c and returns Error (SSE).
    """
    n = len(y_subset)
    if n < 5: return np.inf # Need enough points for a valid physical fit

    # 1. Fit Polynomial of Degree 2 (Parabola)
    # This function solves the Normal Equations (A_T * A * x = A_T * b) internally.
    coeffs = np.polyfit(x_subset, y_subset, 2)

    # 2. Calculate Predicted Values
    # p(x) = ax^2 + bx + c
    p = np.poly1d(coeffs)
    y_pred = p(x_subset)

    # 3. Calculate Sum of Squared Errors
    sse = np.sum((y_subset - y_pred)**2)
    return sse, coeffs

def find_parabolic_limit(k_pts, E_vals):
    """
    Finds the optimal 'Knot' that splits the data into:
    1. A valid Parabolic regime (Low Error)
    2. An invalid Anharmonic regime (High Error)
    """
    n = len(E_vals)
    best_knot = -1
    min_total_sse = np.inf
    best_mass_coeffs = None

    print(f"--- 2. Scanning {n} points for the Parabolic Limit ---")

    # Brute Force Search for the Single Best Split (Bellman Logic for K=2)
    # We ignore the first 10 points (too small) and last 10 (too far)
    for i in range(10, n - 10):

        # Segment 1: Try to fit Parabola to k[0...i]
        sse_1, coeffs_1 = compute_quadratic_sse(k_pts[:i], E_vals[:i])

        # Segment 2: The rest is "garbage" (anharmonic).
        # We model it loosely to account for variance, but focus on S1 error.
        # For this specific physics problem, we mostly care that S1 is perfect.
        # We fit a generic line/poly to the rest to calculate its 'cost'.
        sse_2, _ = compute_quadratic_sse(k_pts[i:], E_vals[i:])

        total_cost = sse_1 + sse_2

        # Bellman Optimality Principle: Minimize Global Error
        if total_cost < min_total_sse:
            min_total_sse = total_cost
            best_knot = i
            best_mass_coeffs = coeffs_1

    return best_knot, best_mass_coeffs

# Run the Algorithm
knot_index, coeffs = find_parabolic_limit(k_axis, E_data)

# Calculate Effective Mass from curvature 'a'
# E = ax^2 + ...  => m* = 1 / (2a)
curvature_a = coeffs[0]
calculated_mass = 1.0 / (2.0 * curvature_a)

print(f"    > Optimal Knot found at index: {knot_index}")
print(f"    > Calculated Effective Mass: {calculated_mass:.4f}")


# ------------------------------------------------------------------------------
# PART 3: VISUALIZATION (Publication Quality)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# 1. Plot Raw Data
plt.scatter(k_axis, E_data, color='lightgray', label='DFT Data (Noisy)', s=25)

# 2. Plot The "Valid Physics" Regime (Red)
k_valid = k_axis[:knot_index]
E_model = np.polyval(coeffs, k_valid)
plt.plot(k_valid, E_model, 'r-', linewidth=3, label=f'Detected Parabolic Region (m*={calculated_mass:.3f})')

# 3. Plot The "Anharmonic" Regime (Blue Dashed)
plt.plot(k_axis[knot_index:], E_data[knot_index:], 'b--', alpha=0.5, label='Anharmonic / Invalid')

# 4. Draw the Decision Boundary
plt.axvline(k_axis[knot_index], color='green', linestyle=':', linewidth=2)
plt.text(k_axis[knot_index], np.max(E_data)*0.8, '  Parabolic Limit', color='green', fontweight='bold')

plt.title("AutoMass: Automated Extraction of Effective Mass", fontsize=14)
plt.xlabel("Momentum ($k$)")
plt.ylabel("Energy ($E$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
