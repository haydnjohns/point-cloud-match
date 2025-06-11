import numpy as np
import matplotlib.pyplot as plt
from pycpd import AffineRegistration

def generate_custom_point_cloud(
    noise_level=0.005,
    x_translation=0.1,
    y_translation=0.15,
    rotation=10,
    num_shared_points=25,
    num_unique_points=5
):
    # Step 1: Generate shared base points
    shared_points = np.random.rand(num_shared_points, 2)

    # Step 2: Add Gaussian noise to create corresponding points in Y
    noise = np.random.normal(scale=noise_level, size=shared_points.shape)
    Y_common = shared_points + noise

    # Step 3: Generate unique (unmatched) points
    X_unique = np.random.rand(num_unique_points, 2)
    Y_unique = np.random.rand(num_unique_points, 2)

    # Combine points
    X = np.vstack([shared_points, X_unique])
    Y = np.vstack([Y_common, Y_unique])

    # Step 4: Apply rotation about the origin
    theta = np.deg2rad(rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    Y_rotated = Y @ R.T

    # Step 5: Apply translation
    translation = np.array([x_translation, y_translation])
    Y_transformed = Y_rotated + translation

    return X, Y_transformed

# Parameters
noise_level = 0.005
x_translation = 0.1
y_translation = 0.1
rotation = 10
num_shared_points = 25
num_unique_points = 5

# Generate data
X, Y = generate_custom_point_cloud(noise_level, x_translation, y_translation, rotation, num_shared_points, num_unique_points)

# Perform registration
reg = AffineRegistration(X=X, Y=Y, w=0.3)
TY, _ = reg.register()

# Print comparisons with % errors
output = True
if output:

    # Affine matrix and translation
    B = reg.B
    t = reg.t

    # Estimate rotation angle
    cos_theta = B[0, 0]
    sin_theta = B[1, 0]
    theta_rad = np.arctan2(sin_theta, cos_theta)
    theta_deg = np.rad2deg(theta_rad)

    # SVD for scale factors
    U, S, Vt = np.linalg.svd(B)

    print(f"\nEstimated rotation angle:     {theta_deg:.2f} degrees")
    print(f"Actual rotation angle:        {rotation:.2f} degrees")
    print(f"Rotation angle error:         {100 * (theta_deg - rotation) / rotation:.0f} %")

    print(f"\nEstimated X Translation:      {t[0]:.4f}")
    print(f"Actual X Translation:         {x_translation:.4f}")
    print(f"X Translation error:          {100 * (t[0] + x_translation) / t[0]:.0f} %")

    print(f"\nEstimated Y Translation:      {t[1]:.4f}")
    print(f"Actual Y Translation:         {y_translation:.4f}")
    print(f"Y Translation error:          {100 * (t[1] + y_translation) / t[1]:.0f} %")

    print(f"\nEstimated Scaling factors:    {S}")
    scale_x_error = 100 * (S[0] - 1.0)
    scale_y_error = 100 * (S[1] - 1.0)
    print(f"X Scale error:                {scale_x_error:.0f} %")
    print(f"Y Scale error:                {scale_y_error:.0f} %")

# Plotting
plot = True
if plot:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(Y[:, 0], Y[:, 1], label='Y (target)', alpha=0.7)
    axs[0].scatter(X[:, 0], X[:, 1], label='X (source)', alpha=0.7)
    axs[0].set_title("Before Registration")
    axs[0].legend()

    axs[1].scatter(TY[:, 0], TY[:, 1], label='TY (aligned)', alpha=0.7)
    axs[1].scatter(X[:, 0], X[:, 1], label='X (source)', alpha=0.7)
    axs[1].set_title("After Registration")
    axs[1].legend()

    for ax in axs:
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()