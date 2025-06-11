import numpy as np
import matplotlib.pyplot as plt
from pycpd import AffineRegistration

def generate_custom_point_cloud_3d(
    noise_level=0.005,
    x_translation=0.1,
    y_translation=0.15,
    z_translation=0.2,
    rotation_z=10,  # Rotation around Z-axis only
    z_scaling=1.1,  # Scaling along Z-axis only
    num_shared_points=25,
    num_unique_points=5
):
    # Step 1: Generate shared base points in 3D
    shared_points = np.random.rand(num_shared_points, 3)

    # Step 2: Add Gaussian noise to create corresponding points in Y
    noise = np.random.normal(scale=noise_level, size=shared_points.shape)
    Y_common = shared_points + noise

    # Step 3: Generate unique (unmatched) points in 3D
    X_unique = np.random.rand(num_unique_points, 3)
    Y_unique = np.random.rand(num_unique_points, 3)

    # Combine points
    X = np.vstack([shared_points, X_unique])
    Y = np.vstack([Y_common, Y_unique])

    # Step 4: Apply rotation around Z-axis
    theta_z = np.deg2rad(rotation_z)
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0,                0,                1]
    ])
    Y_rotated = Y @ R_z.T

    # Step 5: Apply scaling along Z-axis
    S_z = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, z_scaling]
    ])
    Y_scaled = Y_rotated @ S_z.T

    # Step 6: Apply translation
    translation = np.array([x_translation, y_translation, z_translation])
    Y_transformed = Y_scaled + translation

    return X, Y_transformed

# Parameters for 3D
noise_level = 0.005
x_translation = 0.1
y_translation = 0.1
z_translation = 0.2
rotation_z = 10
z_scaling = 1.1
num_shared_points = 25
num_unique_points = 5

# Generate 3D data
X, Y = generate_custom_point_cloud_3d(
    noise_level, x_translation, y_translation, z_translation,
    rotation_z, z_scaling, num_shared_points, num_unique_points
)

# Perform registration
# For 3D, the input to AffineRegistration should be (N, 3)
reg = AffineRegistration(X=X, Y=Y, w=0.3)
TY, _ = reg.register()

# Print comparisons with % errors
output = True
if output:
    # Affine matrix and translation
    B = reg.B
    t = reg.t

    # Estimate rotation angle around Z-axis
    # We are interested in the 2x2 block corresponding to X and Y
    # The rotation around Z will be in the top-left 2x2 submatrix
    cos_theta_z = B[0, 0]
    sin_theta_z = B[1, 0]
    theta_z_rad_estimated = np.arctan2(sin_theta_z, cos_theta_z)
    theta_z_deg_estimated = np.rad2deg(theta_z_rad_estimated)

    # SVD for scale factors
    # For a purely Z-scaled and Z-rotated transformation, the SVD of B
    # should ideally reveal the scale factors.
    # However, B is a 3x3 matrix representing the full affine transformation.
    # To isolate scaling along Z, we observe the B[2,2] component.
    # For a perfect rotation around Z and scaling only along Z, B would look like:
    # [[cos_theta, -sin_theta, 0],
    #  [sin_theta,  cos_theta, 0],
    #  [0,          0,         scale_z]]
    # In practice, due to noise and the nature of the affine registration,
    # B will be a general 3x3 matrix.
    # We can use the singular values to get the overall scaling factors, but
    # specifically isolating Z-scaling needs careful interpretation.
    U, S_estimated, Vt = np.linalg.svd(B)

    print(f"\nEstimated Z-axis rotation angle: {theta_z_deg_estimated:.2f} degrees")
    print(f"Actual Z-axis rotation angle:    {rotation_z:.2f} degrees")
    print(f"Z-axis rotation angle error:     {100 * (theta_z_deg_estimated - rotation_z) / rotation_z:.0f} %" if rotation_z != 0 else "N/A (actual rotation is 0)")

    print(f"\nEstimated X Translation:         {-t[0]:.4f}")
    print(f"Actual X Translation:            {x_translation:.4f}")
    print(f"X Translation error:             {100 * (t[0] + x_translation) / x_translation:.0f} %" if x_translation != 0 else "N/A (actual translation is 0)")

    print(f"\nEstimated Y Translation:         {-t[1]:.4f}")
    print(f"Actual Y Translation:            {y_translation:.4f}")
    print(f"Y Translation error:             {100 * (t[1] + y_translation) / y_translation:.0f} %" if y_translation != 0 else "N/A (actual translation is 0)")

    print(f"\nEstimated Z Translation:         {-t[2]:.4f}")
    print(f"Actual Z Translation:            {z_translation:.4f}")
    print(f"Z Translation error:             {100 * (t[2] + z_translation) / z_translation:.0f} %" if z_translation != 0 else "N/A (actual translation is 0)")

    print(f"\nEstimated Scaling factors (Singular Values of B): {S_estimated}")
    # The third singular value (S_estimated[2]) would ideally correspond to the Z-scaling if no other scaling is present.
    # For more precise Z-scaling, one might look at B[2,2] if the other components are purely rotational.
    estimated_z_scale = B[2,2] # Direct observation from B for Z-scale under specific conditions
    print(f"Estimated Z-Scale: {estimated_z_scale:.4f}")
    print(f"Actual Z-Scale:                   {1/z_scaling:.4f}")
    print(f"Z Scale error:                    {100 * (estimated_z_scale - (1/z_scaling)) / z_scaling:.0f} %" if z_scaling != 0 else "N/A (actual scale is 0 or 1)")

# Plotting (3D plots)
plot = True
if plot:
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], label='Y (target)', alpha=0.7)
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], label='X (source)', alpha=0.7)
    ax1.set_title("Before Registration")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_zlim(-0.2, 1.2)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(TY[:, 0], TY[:, 1], TY[:, 2], label='TY (aligned)', alpha=0.7)
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], label='X (source)', alpha=0.7)
    ax2.set_title("After Registration")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_zlim(-0.2, 1.2)

    plt.tight_layout()
    plt.show()