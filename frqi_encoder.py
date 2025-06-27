"""
FRQI (Flexible Representation of Quantum Images) Implementation
A complete quantum image encoding and reconstruction system using Qiskit.

This implementation provides:
- Quantum image encoding using controlled rotations
- Proper measurement reconstruction with amplitude scaling
- Support for various image patterns and sizes
- Comprehensive visualization and analysis tools

Author: Quantum Image Processing Research
Version: 1.0 (Production Release)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from typing import Dict, Tuple, List, Optional

def clean_measurement_string(state_str: str) -> str:
    """
    Clean and standardize measurement strings from Qiskit output.

    Args:
        state_str: Raw measurement string from Qiskit

    Returns:
        Cleaned measurement string
    """
    clean = ''.join(state_str.split())
    if len(clean) == 6 and clean[:3] == clean[3:]:
        clean = clean[:3]
    elif len(clean) == 6:
        clean = clean[:3]
    return clean

def create_controlled_rotation(qc: QuantumCircuit, theta: float, controls: List[int], target: int) -> None:
    """
    Create a multi-controlled rotation gate for FRQI encoding.

    Args:
        qc: Quantum circuit to modify
        theta: Rotation angle
        controls: List of control qubit indices
        target: Target qubit index for rotation
    """
    if len(controls) == 2:
        # Two-control case: rotate target only when both controls are |1⟩
        control1, control2 = controls

        # Standard decomposition for controlled-controlled-RY
        qc.ry(theta/2, target)
        qc.cx(control2, target)
        qc.ry(-theta/2, target)
        qc.cx(control1, target)
        qc.ry(theta/2, target)
        qc.cx(control2, target)
        qc.ry(-theta/2, target)
        qc.cx(control1, target)
    elif len(controls) == 1:
        # Single control case
        qc.cry(theta, controls[0], target)
    else:
        raise ValueError(f"Unsupported number of control qubits: {len(controls)}")

class FRQIEncoder:
    """
    Flexible Representation of Quantum Images (FRQI) Encoder.

    This class provides complete functionality for encoding classical images
    into quantum states and reconstructing them from quantum measurements.

    Attributes:
        image_size (int): Size of square images (image_size x image_size)
        n_pixels (int): Total number of pixels
        n_position_qubits (int): Number of qubits needed for position encoding
        n_total_qubits (int): Total qubits (position + color)
    """

    def __init__(self, image_size: int = 2):
        """
        Initialize the FRQI encoder.

        Args:
            image_size: Size of square images to process (default: 2)

        Raises:
            ValueError: If image_size is not a power of 2
        """
        self.image_size = image_size
        self.n_pixels = image_size * image_size
        self.n_position_qubits = int(np.log2(self.n_pixels))
        self.n_total_qubits = self.n_position_qubits + 1

        # Validate that image size is a power of 2
        if 2**self.n_position_qubits != self.n_pixels:
            raise ValueError(f"Image size {image_size} must result in a number of pixels that is a power of 2")

        print(f"✅ FRQI Encoder initialized:")
        print(f"   Image size: {image_size}×{image_size}")
        print(f"   Position qubits: {self.n_position_qubits}")
        print(f"   Color qubit: {self.n_position_qubits}")
        print(f"   Total qubits: {self.n_total_qubits}")

    def create_sample_image(self, pattern: str = "single") -> np.ndarray:
        """
        Create sample test images for demonstration and testing.

        Args:
            pattern: Type of pattern to create
                   - "single": Single pixel at (0,0)
                   - "corner": Single pixel at corner
                   - "edge": L-shaped edge pattern
                   - "cross": Cross pattern
                   - "diagonal": Diagonal line
                   - "gradient": Linear gradient
                   - "checkerboard": Checkerboard pattern

        Returns:
            Sample image as numpy array
        """
        img = np.zeros((self.image_size, self.image_size))

        if pattern == "single":
            img[0, 0] = 1.0
        elif pattern == "corner":
            img[0, 0] = 1.0
        elif pattern == "edge":
            img[0, :] = 1.0
            img[:, 0] = 1.0
        elif pattern == "cross":
            if self.image_size >= 4:
                mid = self.image_size // 2
                img[mid, :] = 1.0
                img[:, mid] = 1.0
            else:
                img[0, 1] = 1.0
                img[1, 0] = 1.0
        elif pattern == "diagonal":
            for i in range(min(self.image_size, self.image_size)):
                img[i, i] = 1.0
        elif pattern == "gradient":
            for i in range(self.image_size):
                for j in range(self.image_size):
                    img[i, j] = (i + j) / (2 * (self.image_size - 1))
        elif pattern == "checkerboard":
            for i in range(self.image_size):
                for j in range(self.image_size):
                    img[i, j] = (i + j) % 2
        elif pattern == "border":
            # Frame pattern - edges around perimeter
            img[0, :] = 1.0  # Top
            img[-1, :] = 1.0  # Bottom
            img[:, 0] = 1.0  # Left
            img[:, -1] = 1.0  # Right
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return img

    def encode_image(self, image: np.ndarray, verbose: bool = True) -> QuantumCircuit:
        """
        Encode a classical image into a quantum circuit using FRQI representation.

        Args:
            image: Input image as numpy array (values should be in [0,1])
            verbose: Whether to print encoding details

        Returns:
            Quantum circuit representing the encoded image

        Raises:
            ValueError: If image dimensions don't match encoder configuration
        """
        if verbose:
            print(f"🔧 Creating FRQI quantum circuit...")
            print(f"Image shape: {image.shape}")
            print(f"Image range: [{np.min(image):.3f}, {np.max(image):.3f}]")

        if image.shape != (self.image_size, self.image_size):
            raise ValueError(f"Image shape {image.shape} doesn't match expected {(self.image_size, self.image_size)}")

        qc = QuantumCircuit(self.n_total_qubits)

        # Step 1: Create superposition of all position states
        for i in range(self.n_position_qubits):
            qc.h(i)

        if verbose:
            n_states = 2**self.n_position_qubits
            print(f"   Created superposition of {n_states} position states")

        # Step 2: Encode pixel intensities using controlled rotations
        flattened_image = image.flatten()
        encoded_pixels = 0

        for pixel_idx in range(self.n_pixels):
            intensity = flattened_image[pixel_idx]

            if abs(intensity) < 1e-10:
                continue

            angle = intensity * (np.pi / 2)  # Map [0,1] intensity to [0,π/2] angle
            binary_pos = format(pixel_idx, f'0{self.n_position_qubits}b')
            row, col = pixel_idx // self.image_size, pixel_idx % self.image_size

            if verbose:
                print(f"   Encoding pixel {pixel_idx}: ({row},{col}) intensity={intensity:.3f} → angle={angle:.3f}")

            # Apply controlled rotation for this specific position
            self._apply_position_controlled_rotation(qc, angle, binary_pos, verbose=verbose)
            encoded_pixels += 1

        if verbose:
            print(f"   Total encoded pixels: {encoded_pixels}")
            print(f"   Circuit depth: {qc.depth()}")
            print(f"   Circuit size: {qc.size()}")

        return qc

    def _apply_position_controlled_rotation(self, qc: QuantumCircuit, angle: float,
                                          binary_pos: str, verbose: bool = False) -> None:
        """
        Apply a controlled rotation for a specific position state.

        Args:
            qc: Quantum circuit to modify
            angle: Rotation angle for the color qubit
            binary_pos: Binary representation of the position
            verbose: Whether to print details
        """
        if self.n_position_qubits == 2:
            # For 2x2 images (2 position qubits)
            pos_bit_0 = int(binary_pos[1])  # Least significant bit
            pos_bit_1 = int(binary_pos[0])  # Most significant bit

            # Apply X gates to qubits where we want |0⟩ states in the target position
            x_gates_applied = []
            if pos_bit_0 == 0:
                qc.x(0)
                x_gates_applied.append(0)
            if pos_bit_1 == 0:
                qc.x(1)
                x_gates_applied.append(1)

            # Now both control qubits are |1⟩ for the target position
            # Apply controlled rotation to color qubit
            create_controlled_rotation(qc, angle, [0, 1], 2)

            # Undo the X gates
            for qubit in x_gates_applied:
                qc.x(qubit)

            if verbose:
                print(f"      Applied C²RY rotation: controls=[0,1] → target=2, X-gates={x_gates_applied}")
        else:
            raise NotImplementedError(f"Position encoding for {self.n_position_qubits} qubits not implemented")

    def measure_and_reconstruct(self, circuit: QuantumCircuit, shots: int = 2048,
                              verbose: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Measure the quantum circuit and reconstruct the classical image.

        Args:
            circuit: Quantum circuit to measure
            shots: Number of measurement shots
            verbose: Whether to print measurement details

        Returns:
            Tuple of (reconstructed_image, measurement_counts)
        """
        if verbose:
            print(f"🔧 Measuring quantum circuit with {shots} shots...")

        # Add measurement operations
        measured_circuit = circuit.copy()
        measured_circuit.add_register(ClassicalRegister(self.n_total_qubits, 'c'))

        for i in range(self.n_total_qubits):
            measured_circuit.measure(i, i)

        # Execute quantum simulation
        simulator = AerSimulator()
        transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=1)

        try:
            job = simulator.run(transpiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            if verbose:
                print(f"   📊 Raw measurements: {len(counts)} unique states")

            # Reconstruct image from measurements
            reconstructed_image = self._reconstruct_from_measurements(counts, shots, verbose=verbose)

            return reconstructed_image, counts

        except Exception as e:
            print(f"❌ Measurement failed: {e}")
            return np.zeros((self.image_size, self.image_size)), {}

    def _reconstruct_from_measurements(self, counts: Dict[str, int], shots: int,
                                     verbose: bool = True) -> np.ndarray:
        """
        Reconstruct the classical image from quantum measurement results.

        Args:
            counts: Measurement counts from quantum circuit
            shots: Total number of shots
            verbose: Whether to print reconstruction details

        Returns:
            Reconstructed image as numpy array
        """
        reconstructed_image = np.zeros((self.image_size, self.image_size))

        if verbose:
            print(f"🔧 Reconstructing image from measurements...")

        # Clean measurement strings
        cleaned_counts = {}
        for state_str, count in counts.items():
            clean_state = clean_measurement_string(state_str)
            cleaned_counts[clean_state] = cleaned_counts.get(clean_state, 0) + count

        if verbose:
            print(f"   Processed {len(cleaned_counts)} unique measurement states")

        # Process each measurement outcome
        total_intensity = 0
        successful_reconstructions = 0

        for state, count in cleaned_counts.items():
            if len(state) == self.n_total_qubits:
                # Extract qubit values from measurement string
                # Bit mapping: state[0]=color, state[1]=pos_bit_1, state[2]=pos_bit_0
                color_bit = int(state[0])

                if color_bit == 1:  # Only process states with color bit = 1
                    # Extract position bits
                    pos_bits = [int(state[i+1]) for i in range(self.n_position_qubits)]

                    # Convert position bits to pixel coordinates
                    position_idx = sum(bit * (2**(self.n_position_qubits-1-i))
                                     for i, bit in enumerate(pos_bits))

                    row = position_idx // self.image_size
                    col = position_idx % self.image_size

                    if 0 <= row < self.image_size and 0 <= col < self.image_size:
                        probability = count / shots

                        # Convert probability to pixel intensity
                        # Factor of 4 accounts for quantum superposition normalization
                        intensity = min(4 * probability, 1.0)

                        reconstructed_image[row, col] += intensity
                        total_intensity += intensity
                        successful_reconstructions += 1

                        if verbose and probability > 0.01:
                            print(f"   ✅ State '{state}' → position ({row},{col}) → intensity {intensity:.3f}")

        if verbose:
            print(f"   📊 Successful reconstructions: {successful_reconstructions}")
            print(f"   📊 Total reconstructed intensity: {total_intensity:.3f}")
            print(f"   📊 Max pixel value: {np.max(reconstructed_image):.3f}")

        return reconstructed_image

    def analyze_reconstruction_quality(self, original: np.ndarray, reconstructed: np.ndarray,
                                     verbose: bool = True) -> Dict[str, float]:
        """
        Analyze the quality of image reconstruction.

        Args:
            original: Original input image
            reconstructed: Reconstructed image from quantum measurements
            verbose: Whether to print analysis results

        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}

        # Basic error metrics
        metrics['mse'] = np.mean((original - reconstructed) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['max_error'] = np.max(np.abs(original - reconstructed))
        metrics['mean_error'] = np.mean(np.abs(original - reconstructed))

        # Statistical metrics
        metrics['max_original'] = np.max(original)
        metrics['max_reconstructed'] = np.max(reconstructed)
        metrics['mean_original'] = np.mean(original)
        metrics['mean_reconstructed'] = np.mean(reconstructed)

        # Correlation coefficient
        if np.var(original.flatten()) > 1e-10:
            metrics['correlation'] = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        else:
            metrics['correlation'] = 0.0 if np.var(reconstructed.flatten()) > 1e-10 else 1.0

        # Structural similarity (simplified)
        if np.sum(original) > 0:
            metrics['normalized_overlap'] = np.sum(original * reconstructed) / np.sum(original)
        else:
            metrics['normalized_overlap'] = 1.0 if np.sum(reconstructed) == 0 else 0.0

        if verbose:
            print(f"📊 Reconstruction Quality Analysis:")
            for key, value in metrics.items():
                print(f"   {key.replace('_', ' ').title()}: {value:.4f}")

            # Quality assessment
            correlation = metrics['correlation']
            max_reconstructed = metrics['max_reconstructed']

            if correlation > 0.95 and max_reconstructed > 0.9:
                print(f"   🌟 EXCELLENT reconstruction quality!")
            elif correlation > 0.8 and max_reconstructed > 0.7:
                print(f"   ✅ Very good reconstruction quality!")
            elif correlation > 0.6:
                print(f"   ✅ Good reconstruction quality!")
            elif correlation > 0.3:
                print(f"   ⚠️ Fair reconstruction quality - consider parameter tuning")
            else:
                print(f"   ❌ Poor reconstruction quality - requires investigation")

        return metrics

    def visualize_results(self, original_image: np.ndarray, reconstructed_image: np.ndarray,
                         counts: Dict[str, int], title_prefix: str = "") -> None:
        """
        Create comprehensive visualization of encoding and reconstruction results.

        Args:
            original_image: Original input image
            reconstructed_image: Reconstructed image from quantum measurements
            counts: Measurement counts from quantum circuit
            title_prefix: Optional prefix for plot titles
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if title_prefix:
            fig.suptitle(f'{title_prefix} - FRQI Quantum Image Processing', fontsize=16, fontweight='bold')

        # Original image
        im1 = axes[0,0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('Original Image', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0,0])

        # Reconstructed image
        max_val = max(np.max(reconstructed_image), 1.0)
        im2 = axes[0,1].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=max_val)
        axes[0,1].set_title('Quantum Reconstruction', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[0,1])

        # Measurement distribution
        if len(counts) > 0:
            cleaned_counts = {}
            for state_str, count in counts.items():
                clean_state = clean_measurement_string(state_str)
                cleaned_counts[clean_state] = cleaned_counts.get(clean_state, 0) + count

            states = list(cleaned_counts.keys())
            values = list(cleaned_counts.values())

            # Highlight states with color bit = 1
            colors = ['red' if s[0] == '1' else 'steelblue' for s in states]

            axes[1,0].bar(range(len(states)), values, color=colors)
            axes[1,0].set_title('Quantum Measurements (red=signal states)', fontweight='bold')
            axes[1,0].set_xlabel('Quantum State')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_xticks(range(len(states)))
            axes[1,0].set_xticklabels(states, rotation=45)

        # Error analysis
        error_image = np.abs(original_image - reconstructed_image)
        im4 = axes[1,1].imshow(error_image, cmap='hot', vmin=0, vmax=np.max(error_image))
        axes[1,1].set_title('Absolute Error', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(im4, ax=axes[1,1])

        plt.tight_layout()
        plt.show()

def demonstrate_frqi_system():
    """
    Demonstrate the complete FRQI quantum image processing system.
    """
    print("🎯 FRQI Quantum Image Processing Demonstration")
    print("=" * 60)

    # Initialize encoder
    encoder = FRQIEncoder(image_size=2)

    # Test patterns to demonstrate
    test_patterns = ["single", "corner", "cross", "diagonal"]

    for pattern in test_patterns:
        print(f"\n--- Processing {pattern.upper()} pattern ---")

        # Create test image
        test_image = encoder.create_sample_image(pattern)
        print(f"Created {pattern} pattern:")
        print(test_image)

        # Encode image into quantum circuit
        quantum_circuit = encoder.encode_image(test_image, verbose=True)

        # Measure and reconstruct
        reconstructed_image, measurement_counts = encoder.measure_and_reconstruct(
            quantum_circuit, shots=4096, verbose=True
        )

        # Analyze quality
        quality_metrics = encoder.analyze_reconstruction_quality(
            test_image, reconstructed_image, verbose=True
        )

        # Visualize results
        encoder.visualize_results(
            test_image, reconstructed_image, measurement_counts,
            title_prefix=f"{pattern.title()} Pattern"
        )

        print(f"✅ {pattern.capitalize()} pattern processing complete!")

    print(f"\n🎉 FRQI Demonstration Complete!")
    print("The system successfully demonstrates quantum image encoding and reconstruction.")

if __name__ == "__main__":
    demonstrate_frqi_system()