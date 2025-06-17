"""
FRQI (Flexible Representation of Quantum Images) Encoder
Implements quantum image encoding for quantum image processing algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit import transpile
from PIL import Image
import math

class FRQIEncoder:
    """
    FRQI (Flexible Representation of Quantum Images) implementation
    Encodes classical images into quantum states for quantum image processing
    """

    def __init__(self, image_size=2):
        """
        Initialize FRQI encoder

        Args:
            image_size (int): Size of square image (2^n x 2^n pixels)
        """
        self.image_size = image_size
        self.n_pixels = image_size * image_size
        self.n_position_qubits = int(np.log2(self.n_pixels))
        self.n_total_qubits = self.n_position_qubits + 1  # +1 for color qubit

        print(f"FRQI Encoder initialized:")
        print(f"  Image size: {image_size}x{image_size} pixels")
        print(f"  Total pixels: {self.n_pixels}")
        print(f"  Position qubits needed: {self.n_position_qubits}")
        print(f"  Total qubits needed: {self.n_total_qubits}")

    def create_sample_image(self, pattern="edge"):
        """
        Create sample test images for quantum processing

        Args:
            pattern (str): Type of pattern ("edge", "corner", "cross", "random")

        Returns:
            np.ndarray: 2D image array with values [0,1]
        """
        img = np.zeros((self.image_size, self.image_size))

        if pattern == "edge":
            # Create edge pattern - useful for edge detection testing
            img[0, :] = 1.0  # Top edge
            img[:, 0] = 1.0  # Left edge

        elif pattern == "corner":
            # Single bright pixel in corner
            img[0, 0] = 1.0

        elif pattern == "cross":
            # Cross pattern
            if self.image_size >= 4:
                mid = self.image_size // 2
                img[mid, :] = 1.0  # Horizontal line
                img[:, mid] = 1.0  # Vertical line
            else:
                img[0, 1] = 1.0
                img[1, 0] = 1.0

        elif pattern == "random":
            # Random pattern
            np.random.seed(42)  # For reproducibility
            img = np.random.rand(self.image_size, self.image_size)

        return img

    def normalize_image(self, image):
        """
        Normalize image values to [0, π/2] for FRQI encoding

        Args:
            image (np.ndarray): Input image with values [0,1]

        Returns:
            np.ndarray: Normalized angles for FRQI encoding
        """
        # Normalize to [0, π/2] range for quantum amplitudes
        angles = image * (np.pi / 2)
        return angles

    def create_frqi_circuit(self, image_angles):
        """
        Create FRQI quantum circuit from image angles

        Args:
            image_angles (np.ndarray): Normalized image angles

        Returns:
            QuantumCircuit: FRQI encoded quantum circuit
        """
        # Create quantum registers
        position_qubits = QuantumRegister(self.n_position_qubits, 'pos')
        color_qubit = QuantumRegister(1, 'color')
        classical_bits = ClassicalRegister(self.n_total_qubits, 'c')

        # Create circuit
        circuit = QuantumCircuit(position_qubits, color_qubit, classical_bits)

        # Step 1: Create superposition of all position states
        for i in range(self.n_position_qubits):
            circuit.h(position_qubits[i])

        # Step 2: Apply controlled rotations based on position
        flattened_angles = image_angles.flatten()

        for pixel_idx in range(self.n_pixels):
            # Convert pixel index to binary representation for position qubits
            binary_pos = format(pixel_idx, f'0{self.n_position_qubits}b')

            # Get the angle for this pixel
            angle = flattened_angles[pixel_idx]

            # Skip if angle is zero (no rotation needed)
            if abs(angle) < 1e-10:
                continue

            # For 2x2 image (2 position qubits), we need controlled rotations
            if self.n_position_qubits == 2:
                # Handle each position combination using mcry (multi-controlled RY)
                control_qubits = []

                # Set up control qubits based on binary position
                for bit_idx, bit_val in enumerate(binary_pos):
                    if bit_val == '0':
                        # Apply X gate to flip qubit for control on |0⟩
                        circuit.x(position_qubits[bit_idx])
                    control_qubits.append(position_qubits[bit_idx])

                # Apply multi-controlled RY rotation
                circuit.mcry(2 * angle, control_qubits, color_qubit[0])

                # Undo X gates applied for control
                for bit_idx, bit_val in enumerate(binary_pos):
                    if bit_val == '0':
                        circuit.x(position_qubits[bit_idx])

            elif self.n_position_qubits == 1:
                # For 1 position qubit (2x1 or 1x2 image)
                if binary_pos == "0":
                    circuit.x(position_qubits[0])
                    circuit.cry(2 * angle, position_qubits[0], color_qubit[0])
                    circuit.x(position_qubits[0])
                else:  # binary_pos == "1"
                    circuit.cry(2 * angle, position_qubits[0], color_qubit[0])

            else:
                # For cases with more than 2 position qubits
                control_qubits = []
                x_applied = []

                for bit_idx, bit_val in enumerate(binary_pos):
                    if bit_val == '0':
                        circuit.x(position_qubits[bit_idx])
                        x_applied.append(bit_idx)
                    control_qubits.append(position_qubits[bit_idx])

                # Apply multi-controlled rotation
                circuit.mcry(2 * angle, control_qubits, color_qubit[0])

                # Undo X gates
                for bit_idx in x_applied:
                    circuit.x(position_qubits[bit_idx])

        return circuit

    def encode_image(self, image):
        """
        Complete FRQI encoding pipeline

        Args:
            image (np.ndarray): Input image

        Returns:
            QuantumCircuit: FRQI encoded quantum circuit
        """
        print("Encoding image with FRQI...")
        print(f"Original image shape: {image.shape}")
        print(f"Original image:\n{image}")

        # Normalize image to quantum angles
        angles = self.normalize_image(image)
        print(f"Normalized angles (radians):\n{angles}")

        # Create FRQI quantum circuit
        circuit = self.create_frqi_circuit(angles)

        print(f"FRQI circuit created with {circuit.num_qubits} qubits and depth {circuit.depth()}")
        return circuit

    def measure_and_reconstruct(self, circuit, shots=1024):
        """
        Measure FRQI circuit and reconstruct classical image

        Args:
            circuit (QuantumCircuit): FRQI encoded circuit
            shots (int): Number of measurement shots

        Returns:
            tuple: (reconstructed_image, measurement_counts)
        """
        # Add measurements
        measured_circuit = circuit.copy()
        measured_circuit.measure_all()

        # Run on simulator
        simulator = AerSimulator()
        compiled_circuit = transpile(measured_circuit, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Reconstruct image from measurement statistics
        reconstructed_image = np.zeros((self.image_size, self.image_size))

        for state, count in counts.items():
            # Clean the state string (remove spaces)
            clean_state = state.replace(' ', '')

            # Parse measurement result: last bit is color, rest are position
            color_bit = int(clean_state[0])  # Color qubit result
            position_bits = clean_state[1:]   # Position qubits result

            # Convert position bits to pixel coordinates
            if position_bits:  # Make sure we have position bits
                position_idx = int(position_bits, 2)
            row = position_idx // self.image_size
            col = position_idx % self.image_size

            # Make sure coordinates are valid
            if 0 <= row < self.image_size and 0 <= col < self.image_size:
                # Color intensity from measurement probability
                if color_bit == 1:  # Measured |1⟩ state
                    reconstructed_image[row, col] += count / shots

        return reconstructed_image, counts

    def visualize_results(self, original_image, reconstructed_image, counts):
        """
        Visualize original vs reconstructed image and measurement statistics

        Args:
            original_image (np.ndarray): Original classical image
            reconstructed_image (np.ndarray): Reconstructed from quantum measurements
            counts (dict): Measurement count statistics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        im1 = axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original Image')
        axes[0].set_xlabel('Pixel Column')
        axes[0].set_ylabel('Pixel Row')
        plt.colorbar(im1, ax=axes[0])

        # Reconstructed image
        im2 = axes[1].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Reconstructed from Quantum Measurements')
        axes[1].set_xlabel('Pixel Column')
        axes[1].set_ylabel('Pixel Row')
        plt.colorbar(im2, ax=axes[1])

        # Measurement statistics
        axes[2].bar(range(len(counts)), list(counts.values()))
        axes[2].set_title('Quantum Measurement Counts')
        axes[2].set_xlabel('Quantum State')
        axes[2].set_ylabel('Count')
        axes[2].set_xticks(range(len(counts)))
        axes[2].set_xticklabels([state[::-1] for state in counts.keys()], rotation=45)

        plt.tight_layout()
        plt.show()

        # Calculate reconstruction fidelity
        mse = np.mean((original_image - reconstructed_image) ** 2)
        print(f"Reconstruction Mean Squared Error: {mse:.4f}")

        return mse

# Example usage and testing
if __name__ == "__main__":
    # Test FRQI encoding with different image patterns
    print("=== FRQI Quantum Image Encoding Demo ===\n")

    # Initialize encoder for 2x2 images (needs 3 qubits total)
    encoder = FRQIEncoder(image_size=2)

    # Test different patterns
    patterns = ["edge", "corner", "cross"]

    for pattern in patterns:
        print(f"\n--- Testing {pattern.upper()} pattern ---")

        # Create test image
        test_image = encoder.create_sample_image(pattern)
        print(f"Test image ({pattern}):\n{test_image}")

        # Encode image
        frqi_circuit = encoder.encode_image(test_image)

        # Print circuit info
        print(f"Circuit depth: {frqi_circuit.depth()}")
        print(f"Circuit gates: {frqi_circuit.count_ops()}")

        # Measure and reconstruct
        reconstructed, counts = encoder.measure_and_reconstruct(frqi_circuit, shots=1024)

        # Visualize results
        mse = encoder.visualize_results(test_image, reconstructed, counts)

        print(f"Reconstruction quality (MSE): {mse:.4f}")

    print("\n=== FRQI encoding demonstration complete ===")
    print("Next step: Implement quantum edge detection on FRQI-encoded images!")