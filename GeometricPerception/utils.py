import numpy as np


def random_rotation_matrix() -> np.ndarray:
    """
    Generate a random 3D rotation matrix from a uniformly sampled unit quaternion.
    
    Returns:
        R: 3x3 rotation matrix (SO(3))
    """
    # Sample random quaternion uniformly on the 4-sphere
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    
    # Convert uniform samples to quaternion [w, x, y, z]
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    
    # Convert quaternion to rotation matrix
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Generate a 3D rotation matrix from axis-angle representation.
    
    Args:
        axis: 3D unit vector representing the rotation axis
        angle: Rotation angle in radians
        
    Returns:
        R: 3x3 rotation matrix (SO(3))
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R


def random_rotation_matrix_axis_angle() -> np.ndarray:
    """
    Generate a random 3D rotation matrix using axis-angle representation.
    
    Returns:
        R: 3x3 rotation matrix (SO(3))
    """
    # Random axis: sample from unit sphere
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # Random angle uniformly from [0, pi]
    # Note: for uniform rotations, angle should be sampled with density sin(angle/2)^2
    # but uniform [0, pi] is a common simple choice
    angle = np.random.uniform(0, np.pi)
    
    return rotation_matrix_from_axis_angle(axis, angle)


def project_to_SO3(M: np.ndarray) -> np.ndarray:
    """
    Project a 3x3 matrix onto SO(3) using SVD.
    
    Finds the closest rotation matrix R to M in Frobenius norm:
        R = argmin_R ||M - R||_F  subject to R ∈ SO(3)
    
    The solution is R = U @ V^T where M = U @ S @ V^T is the SVD,
    with a correction to ensure det(R) = +1.
    
    Args:
        M: 3x3 matrix to project
        
    Returns:
        R: 3x3 rotation matrix in SO(3)
    """
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    # Ensure det(R) = +1 (not a reflection)
    if np.linalg.det(R) < 0:
        # Flip sign of last column of U
        U[:, -1] *= -1
        R = U @ Vt
    
    return R


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute the geodesic (angular) error between two rotation matrices.
    
    This is the angle of the rotation needed to go from R1 to R2,
    computed as: theta = arccos((trace(R1^T @ R2) - 1) / 2)
    
    Args:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix
        
    Returns:
        error: Rotation error in radians, in range [0, pi]
    """
    # Compute relative rotation
    R_rel = R1.T @ R2
    
    # Compute trace
    trace = np.trace(R_rel)
    
    # Clamp to handle numerical errors
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    
    # Geodesic distance (rotation angle)
    error = np.arccos(cos_angle)
    
    return error


def rotation_error_degrees(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute the geodesic (angular) error between two rotation matrices in degrees.
    
    Args:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix
        
    Returns:
        error: Rotation error in degrees, in range [0, 180]
    """
    return np.rad2deg(rotation_error(R1, R2))


def generate_random_pnp_problem(
    N: int,
    fov_degrees: float = 120.0,
    z_min: float = 1.0,
    z_max: float = 10.0,
    noise_std: float = 0.01,
    translation_scale: float = 5.0
) -> dict:
    """
    Generate a random Perspective-n-Point (PnP) problem.
    
    The setup:
    - Camera is at origin, looking down the +z axis
    - N 3D points are generated in front of the camera within the field of view
    - Points are projected to 2D with added Gaussian noise
    - Points are transformed to world frame via random rotation and translation
    
    Args:
        N: Number of 3D points to generate
        fov_degrees: Field of view cone angle in degrees (default 120)
        z_min: Minimum depth of points (default 1.0)
        z_max: Maximum depth of points (default 10.0)
        noise_std: Standard deviation of Gaussian noise on 2D projections (default 0.01)
        translation_scale: Scale for random translation (default 5.0)
        
    Returns:
        dict containing:
            - 'R_gt': 3x3 rotation matrix (world to camera, i.e., R such that p_cam = R @ p_world + t)
            - 't_gt': 3x1 translation vector (world to camera)
            - 'points_3d_world': Nx3 array of 3D points in world frame
            - 'points_2d_noisy': Nx2 array of noisy 2D projections (normalized coordinates)
            - 'points_2d_clean': Nx2 array of clean 2D projections (for reference)
            - 'points_3d_camera': Nx3 array of 3D points in camera frame (for reference)
    """
    # Convert FOV to half-angle in radians
    half_fov_rad = np.deg2rad(fov_degrees / 2)
    
    # Generate N random 3D points in camera frame within the FOV cone
    # For each point: sample depth z, then sample x, y within the cone at that depth
    z_vals = np.random.uniform(z_min, z_max, N)
    
    # Maximum radius at each depth for the cone
    max_radius = z_vals * np.tan(half_fov_rad)
    
    # Sample points uniformly within a disk at each depth
    # Use sqrt for uniform sampling in disk
    r = np.sqrt(np.random.uniform(0, 1, N)) * max_radius
    theta = np.random.uniform(0, 2 * np.pi, N)
    
    x_vals = r * np.cos(theta)
    y_vals = r * np.sin(theta)
    
    # 3D points in camera frame (Nx3)
    points_3d_camera = np.column_stack([x_vals, y_vals, z_vals])
    
    # Project to 2D (normalized image coordinates)
    points_2d_clean = points_3d_camera[:, :2] / points_3d_camera[:, 2:3]
    
    # Add Gaussian noise to 2D projections
    noise = np.random.normal(0, noise_std, points_2d_clean.shape)
    points_2d_noisy = points_2d_clean + noise
    
    # Generate random rotation (camera to world)
    R_cam_to_world = random_rotation_matrix()

    # Generate random unit direction for translation
    dir_vec = np.random.randn(3)
    dir_vec /= np.linalg.norm(dir_vec)
    # Generate random translation norm between 0 and translation_scale
    t_scale = np.random.uniform(0, translation_scale)
    # Camera-to-world translation (for transforming points)
    t_cam_to_world = t_scale * dir_vec

    # Transform points to world frame: p_world = R_cam_to_world @ p_camera + t_cam_to_world
    points_3d_world = (R_cam_to_world @ points_3d_camera.T).T + t_cam_to_world

    # Ground truth: R_gt (world to camera) and t_gt
    R_gt = R_cam_to_world.T
    t_gt = -R_gt @ t_cam_to_world
    
    return {
        'R_gt': R_gt,
        't_gt': t_gt,
        'points_3d_world': points_3d_world,
        'points_2d_noisy': points_2d_noisy,
        'points_2d_clean': points_2d_clean,
        'points_3d_camera': points_3d_camera
    }


if __name__ == "__main__":
    # Demo rotation functions
    R1 = random_rotation_matrix()
    R2 = random_rotation_matrix()
    
    print("=" * 60)
    print("Rotation Matrix Demo")
    print("=" * 60)
    print("\nR1 (from quaternion):")
    print(R1)
    print(f"\nR1 is valid rotation: det = {np.linalg.det(R1):.6f}, R^T R = I check: {np.allclose(R1.T @ R1, np.eye(3))}")
    
    print("\nR2 (from quaternion):")
    print(R2)
    
    error = rotation_error(R1, R2)
    print(f"\nRotation error between R1 and R2: {error:.4f} rad ({np.rad2deg(error):.2f} deg)")
    
    # Self-check: error with itself should be 0
    self_error = rotation_error(R1, R1)
    print(f"Self-error (should be 0): {self_error:.10f} rad")
    
    # Demo PnP problem generation
    print("\n" + "=" * 60)
    print("PnP Problem Generation Demo")
    print("=" * 60)
    
    pnp = generate_random_pnp_problem(N=10, noise_std=0.02)
    
    print(f"\nGenerated PnP problem with {len(pnp['points_3d_world'])} points")
    print(f"\nGroundtruth rotation R_gt (world to camera):")
    print(pnp['R_gt'])
    print(f"\nGroundtruth translation t_gt: {pnp['t_gt']}")
    
    print(f"\n3D points in world frame (first 3):")
    print(pnp['points_3d_world'][:3])
    
    print(f"\n2D noisy projections (first 3):")
    print(pnp['points_2d_noisy'][:3])
    
    # Verify: transform world points back to camera and check projection
    points_cam_reconstructed = (pnp['R_gt'] @ pnp['points_3d_world'].T).T + pnp['t_gt']
    proj_reconstructed = points_cam_reconstructed[:, :2] / points_cam_reconstructed[:, 2:3]
    
    print(f"\nVerification - max projection error (clean): {np.max(np.abs(proj_reconstructed - pnp['points_2d_clean'])):.2e}")

