import numpy as np

def calculate_joint_angle(joint_a, joint_b, joint_c):
    # Convert joint coordinates to vectors
    vector_ab = np.array(joint_b) - np.array(joint_a)
    vector_ac = np.array(joint_c) - np.array(joint_a)

    # Calculate the angle using the dot product
    cos_angle = np.dot(vector_ab, vector_ac) / (np.linalg.norm(vector_ab) * np.linalg.norm(vector_ac))
    angle = np.arccos(cos_angle)  # Angle in radians

    return np.degrees(angle)  # Convert to degrees

# Example usage
knee_angle = calculate_joint_angle([0, 0, 0], [1, 0, 0], [1, 1, 0])
print(f"Knee Angle: {knee_angle} degrees")