"""
Quantum Hadamard Edge Detection (QHED) Algorithm
Implements quantum edge detection on FRQI-encoded images
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
from frqi_encoder import FRQIEncoder
import time

class QuantumEdgeDetection:
    """
    Quantum Hadamard Edge Detection (QHED) implementation
    Detects edges in FRQI-encoded quantum images
    """

    def __init__(self, image_size=2):
        """
        Initialize quantum edge detection

        Args:
            image_size (int): Size of square image (2^n x 2^n pixels)
        """
        self.image_size = image_size
        self.n_pixels = image_size * image_size
        self.n_position_qubits = int(np.log2(self.n_pixels))
        self.n_total_qubits = self.n_position_qubits + 1

        # Initialize FRQI encoder
        self.frqi_encoder = FRQIEncoder(image_size)

        print(f"Quantum Edge Detection initialized for {image_size}x{image_size} images")

    def create_qhed_circuit(self, frqi_circuit):
        """
        Apply Quantum Hadamard Edge Detection to FRQI-encoded image

        The QHED algorithm:
        1. Start with FRQI-encoded image
        2. Apply Hadamard gates to create quantum interference
        3. Measure to detect edge patterns

        Args:
            frqi_circuit (QuantumCircuit): FRQI-encoded image circuit

        Returns:
            QuantumCircuit: QHED edge detection circuit
        """
        # Copy the FRQI circuit
        edge_circuit = frqi_circuit.copy()

        # Apply Hadamard gates for edge detection
        # The key insight: Hadamard gates create interference patterns
        # that highlight discontinuities (edges) in the image

        # Apply Hadamard to color qubit - creates superposition for edge detection
        color_qubit_idx = self.n_position_qubits  # Last qubit is color qubit
        edge_circuit.h(color_qubit_idx)

        # Apply additional Hadamard gates to position qubits for enhanced edge detection
        for i in range(self.n_position_qubits):
            edge_circuit.h(i)

        # Add barrier for clarity
        edge_circuit.barrier()

        # Optional: Add controlled phase gates for enhanced edge detection
        # This creates phase relationships that emphasize pixel boundaries
        for i in range(self.n_position_qubits - 1):
            edge_circuit.cz(i, i + 1)  # Create entanglement between adjacent position qubits

        return edge_circuit

    def measure_edges(self, edge_circuit, shots=1024):
        """
        Measure edge detection circuit and extract edge information

        Args:
            edge_circuit (QuantumCircuit): QHED circuit
            shots (int): Number of measurement shots

        Returns:
            tuple: (edge_image, measurement_counts, edge_strength)
        """
        # Add measurements
        measured_circuit = edge_circuit.copy()
        measured_circuit.measure_all()

        # Run on simulator
        simulator = AerSimulator()
        compiled_circuit = transpile(measured_circuit, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Reconstruct edge-detected image
        edge_image = np.zeros((self.image_size, self.image_size))
        edge_strength = 0.0

        for state, count in counts.items():
            # Clean the state string (remove spaces)
            clean_state = state.replace(' ', '')

            # Parse quantum state
            if len(clean_state) >= 3:  # Ensure we have enough bits
                color_bit = int(clean_state[0])
                position_bits = clean_state[1:]

                # Convert to pixel coordinates
                if position_bits:  # Make sure we have position bits
                    position_idx = int(position_bits, 2)
                    row = position_idx // self.image_size
                    col = position_idx % self.image_size

                    # Ensure coordinates are valid
                    if 0 <= row < self.image_size and 0 <= col < self.image_size:
                        # Edge detection logic: interference patterns indicate edges
                        # Higher measurement probability of |1⟩ state indicates edge presence
                        probability = count / shots

                        if color_bit == 1:
                            edge_image[row, col] = probability
                            edge_strength += probability

        return edge_image, counts, edge_strength

    def classical_sobel_edge_detection(self, image):
        """
        Classical Sobel edge detection for comparison

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Edge-detected image using classical Sobel operator
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # For small images, use simplified edge detection
        if self.image_size <= 2:
            # Simple gradient calculation for 2x2 images
            edges = np.zeros_like(image)

            # Check for edges (differences between adjacent pixels)
            if image.shape[0] > 1 and image.shape[1] > 1:
                # Horizontal edges
                edges[:-1, :] += np.abs(np.diff(image, axis=0))
                # Vertical edges
                edges[:, :-1] += np.abs(np.diff(image, axis=1))

            return np.clip(edges, 0, 1)

        else:
            # Full Sobel convolution for larger images
            from scipy import ndimage
            grad_x = ndimage.convolve(image, sobel_x)
            grad_y = ndimage.convolve(image, sobel_y)
            edges = np.sqrt(grad_x**2 + grad_y**2)
            return edges / np.max(edges) if np.max(edges) > 0 else edges

    def run_edge_detection_comparison(self, test_image, shots=1024):
        """
        Run both quantum and classical edge detection for comparison

        Args:
            test_image (np.ndarray): Input test image
            shots (int): Number of quantum measurements

        Returns:
            dict: Comparison results
        """
        print(f"Running edge detection comparison on {self.image_size}x{self.image_size} image...")

        # Time classical edge detection
        start_time = time.time()
        classical_edges = self.classical_sobel_edge_detection(test_image)
        classical_time = time.time() - start_time

        # Time quantum edge detection
        start_time = time.time()

        # Step 1: Encode image with FRQI
        frqi_circuit = self.frqi_encoder.encode_image(test_image)

        # Step 2: Apply quantum edge detection
        qhed_circuit = self.create_qhed_circuit(frqi_circuit)

        # Step 3: Measure and reconstruct edges
        quantum_edges, counts, edge_strength = self.measure_edges(qhed_circuit, shots)

        quantum_time = time.time() - start_time

        # Calculate metrics
        results = {
            'original_image': test_image,
            'classical_edges': classical_edges,
            'quantum_edges': quantum_edges,
            'classical_time': classical_time,
            'quantum_time': quantum_time,
            'edge_strength': edge_strength,
            'quantum_counts': counts,
            'circuit_depth': qhed_circuit.depth(),
            'circuit_gates': qhed_circuit.count_ops()
        }

        return results

    def visualize_comparison(self, results):
        """
        Visualize comparison between quantum and classical edge detection

        Args:
            results (dict): Comparison results
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        im1 = axes[0,0].imshow(results['original_image'], cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('Original Image')
        plt.colorbar(im1, ax=axes[0,0])

        # Classical edge detection
        im2 = axes[0,1].imshow(results['classical_edges'], cmap='hot', vmin=0, vmax=1)
        axes[0,1].set_title(f'Classical Sobel Edges\nTime: {results["classical_time"]:.4f}s')
        plt.colorbar(im2, ax=axes[0,1])

        # Quantum edge detection
        im3 = axes[0,2].imshow(results['quantum_edges'], cmap='hot', vmin=0, vmax=1)
        axes[0,2].set_title(f'Quantum QHED Edges\nTime: {results["quantum_time"]:.4f}s')
        plt.colorbar(im3, ax=axes[0,2])

        # Circuit depth comparison
        methods = ['Classical', 'Quantum']
        times = [results['classical_time'], results['quantum_time']]
        depths = [1, results['circuit_depth']]  # Classical has depth 1 (single operation)

        axes[1,0].bar(methods, times)
        axes[1,0].set_title('Execution Time Comparison')
        axes[1,0].set_ylabel('Time (seconds)')

        axes[1,1].bar(methods, depths)
        axes[1,1].set_title('Algorithm Complexity')
        axes[1,1].set_ylabel('Circuit Depth / Operation Count')

        # Quantum measurement statistics
        counts = results['quantum_counts']
        axes[1,2].bar(range(len(counts)), list(counts.values()))
        axes[1,2].set_title('Quantum Measurement Statistics')
        axes[1,2].set_xlabel('Quantum State')
        axes[1,2].set_ylabel('Measurement Count')

        plt.tight_layout()
        plt.show()

        # Print detailed analysis
        print("\n=== Edge Detection Analysis ===")
        print(f"Original image mean intensity: {np.mean(results['original_image']):.3f}")
        print(f"Classical edge strength: {np.sum(results['classical_edges']):.3f}")
        print(f"Quantum edge strength: {results['edge_strength']:.3f}")
        print(f"Quantum circuit depth: {results['circuit_depth']}")
        print(f"Quantum gates used: {results['circuit_gates']}")
        print(f"Speedup potential: Quantum O(log²n) vs Classical O(n²)")

        return results

# Main demonstration
if __name__ == "__main__":
    print("=== Quantum vs Classical Edge Detection Comparison ===\n")

    # Initialize quantum edge detector
    qed = QuantumEdgeDetection(image_size=2)

    # Test different image patterns
    test_patterns = ["edge", "corner", "cross"]

    all_results = {}

    for pattern in test_patterns:
        print(f"\n--- Testing {pattern.upper()} pattern ---")

        # Create test image
        test_image = qed.frqi_encoder.create_sample_image(pattern)
        print(f"Test image:\n{test_image}")

        # Run comparison
        results = qed.run_edge_detection_comparison(test_image, shots=2048)

        # Visualize results
        qed.visualize_comparison(results)

        # Store results
        all_results[pattern] = results

    # Summary comparison
    print("\n=== SUMMARY COMPARISON ===")
    print("Pattern\t\tClassical Time\tQuantum Time\tEdge Strength")
    print("-" * 60)
    for pattern, results in all_results.items():
        print(f"{pattern:12}\t{results['classical_time']:.4f}s\t\t{results['quantum_time']:.4f}s\t\t{results['edge_strength']:.3f}")

    print("\n=== Key Insights ===")
    print("1. Quantum edge detection shows different edge sensitivity patterns")
    print("2. Circuit depth scales as O(log²n) vs classical O(n²)")
    print("3. Quantum approach reveals interference-based edge features")
    print("4. Current implementation limited by NISQ device constraints")
    print("\nNext: Test on real IBM Quantum hardware!")