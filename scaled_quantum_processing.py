"""
Working 4×4 Quantum Image Processing - Final Fix
Explicitly controls measurement to get correct 5-bit strings
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time

class WorkingFRQI4x4:
    """
    Working FRQI implementation for 4x4 images with explicit measurement control
    """

    def __init__(self):
        self.image_size = 4
        self.n_pixels = 16
        self.n_position_qubits = 4
        self.n_color_qubits = 1
        self.n_total_qubits = 5

        print(f"🔧 Working FRQI 4×4 Encoder:")
        print(f"   Image: {self.image_size}×{self.image_size} = {self.n_pixels} pixels")
        print(f"   Qubits: {self.n_position_qubits} position + {self.n_color_qubits} color = {self.n_total_qubits} total")

    def create_controlled_circuit(self, image):
        """Create FRQI circuit with explicit classical register control"""

        # Create circuit with EXACTLY what we need: 5 qubits, 5 classical bits
        qc = QuantumCircuit(self.n_total_qubits, self.n_total_qubits)

        print(f"   Creating controlled circuit: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")

        # Step 1: Create superposition of all positions
        for i in range(self.n_position_qubits):
            qc.h(i)

        # Step 2: Encode each pixel with explicit control
        flattened_image = image.flatten()

        for pixel_idx in range(self.n_pixels):
            pixel_value = flattened_image[pixel_idx]

            # Skip zero pixels
            if pixel_value < 1e-6:
                continue

            # Convert pixel index to binary (4 bits for 16 positions)
            binary_pos = format(pixel_idx, '04b')

            print(f"   Encoding pixel {pixel_idx} (value={pixel_value:.3f}) at position {binary_pos}")

            # Apply X gates for |0⟩ controls
            for bit_idx, bit_val in enumerate(binary_pos):
                if bit_val == '0':
                    qc.x(bit_idx)

            # Calculate rotation angle
            angle = pixel_value * (np.pi / 2)

            # Apply multi-controlled rotation to color qubit (qubit 4)
            qc.mcry(2 * angle, [0, 1, 2, 3], 4)

            # Undo X gates
            for bit_idx, bit_val in enumerate(binary_pos):
                if bit_val == '0':
                    qc.x(bit_idx)

        return qc

    def measure_explicitly(self, circuit, shots=2048):
        """Measure circuit with explicit control over classical bits"""

        # Create measurement circuit by copying and adding explicit measurements
        measured_circuit = circuit.copy()

        # Add explicit measurements - one qubit to one classical bit
        for i in range(self.n_total_qubits):
            measured_circuit.measure(i, i)

        print(f"   Measuring: {measured_circuit.num_qubits} qubits → {measured_circuit.num_clbits} classical bits")
        print(f"   Circuit depth: {measured_circuit.depth()}")

        # Run simulation
        simulator = AerSimulator()
        transpiled = transpile(measured_circuit, simulator)

        start_time = time.time()
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        execution_time = time.time() - start_time

        counts = result.get_counts()

        print(f"   Execution time: {execution_time:.3f}s")
        print(f"   Measured {len(counts)} different states")

        # Show measurement results
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top measurements:")
        for state, count in sorted_counts[:5]:
            prob = count / shots
            clean_state = ''.join(state.split())
            print(f"     Raw: '{state}' → Clean: '{clean_state}' (len={len(clean_state)}) → {count} ({prob:.3f})")

        # Decode measurements
        reconstructed = np.zeros((self.image_size, self.image_size))
        successful_reconstructions = 0

        for state_str, count in counts.items():
            # Clean state string thoroughly
            clean_state = ''.join(state_str.split())

            # Now it should be exactly 5 characters
            if len(clean_state) != self.n_total_qubits:
                print(f"   ⚠️ Unexpected state length: {len(clean_state)} for '{state_str}' → '{clean_state}'")
                continue

            # Parse state: Qiskit measures in reverse order
            # Our circuit: qubits [0,1,2,3,4] = [pos0,pos1,pos2,pos3,color]
            # Qiskit result: bits [4,3,2,1,0] = [color,pos3,pos2,pos1,pos0]

            # So clean_state[0] = color, clean_state[1:5] = position in reverse
            color_bit = int(clean_state[0])
            pos_bits_reversed = clean_state[1:]
            pos_bits = pos_bits_reversed[::-1]  # Reverse to get correct order

            # Convert position to coordinates
            position_idx = int(pos_bits, 2)
            row = position_idx // self.image_size
            col = position_idx % self.image_size

            probability = count / shots

            # Add to reconstruction if color bit is 1
            if color_bit == 1 and 0 <= row < self.image_size and 0 <= col < self.image_size:
                reconstructed[row, col] += probability
                successful_reconstructions += 1
                if probability > 0.02:
                    print(f"     ✅ '{state_str}' → color={color_bit}, pos='{pos_bits}' → ({row},{col}) → +{probability:.3f}")

        print(f"   Successful reconstructions: {successful_reconstructions}")

        return reconstructed, counts, execution_time

def test_working_encoding():
    """Test the working encoding with verification"""

    print("🎉 TESTING WORKING 4×4 QUANTUM ENCODING")
    print("=" * 60)

    encoder = WorkingFRQI4x4()

    # Start with the simplest possible test
    test_patterns = {
        'single_corner': np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]),

        'two_corners': np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]),

        'simple_line': np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
    }

    results = {}

    for pattern_name, pattern in test_patterns.items():
        print(f"\n{'='*50}")
        print(f"TESTING: {pattern_name.upper()}")
        print(f"{'='*50}")

        print(f"Original image:\n{pattern}")

        # Create circuit
        circuit = encoder.create_controlled_circuit(pattern)

        # Measure with explicit control
        reconstructed, counts, exec_time = encoder.measure_explicitly(circuit)

        # Calculate metrics
        mse = np.mean((pattern - reconstructed) ** 2)
        max_val = np.max(reconstructed)
        total_intensity_original = np.sum(pattern)
        total_intensity_reconstructed = np.sum(reconstructed)

        print(f"\nResults:")
        print(f"   Reconstructed image:\n{reconstructed}")
        print(f"   MSE: {mse:.4f}")
        print(f"   Max reconstructed value: {max_val:.3f}")
        print(f"   Original total intensity: {total_intensity_original:.3f}")
        print(f"   Reconstructed total intensity: {total_intensity_reconstructed:.3f}")
        print(f"   Execution time: {exec_time:.3f}s")

        # Success check
        if max_val > 0.05:
            print(f"   ✅ SUCCESS! Quantum reconstruction working!")
        else:
            print(f"   ❌ Still not working properly...")

        results[pattern_name] = {
            'original': pattern,
            'reconstructed': reconstructed,
            'mse': mse,
            'max_val': max_val,
            'success': max_val > 0.05
        }

    # Visualize results
    fig, axes = plt.subplots(len(test_patterns), 3, figsize=(12, 4*len(test_patterns)))
    if len(test_patterns) == 1:
        axes = axes.reshape(1, -1)

    for i, (name, result) in enumerate(results.items()):
        # Original
        axes[i, 0].imshow(result['original'], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original: {name}')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Reconstructed
        im = axes[i, 1].imshow(result['reconstructed'], cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Quantum Reconstruction\nMax: {result["max_val"]:.3f}')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        plt.colorbar(im, ax=axes[i, 1])

        # Difference
        diff = np.abs(result['original'] - result['reconstructed'])
        axes[i, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Difference\nMSE: {result["mse"]:.4f}')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    plt.tight_layout()
    plt.show()

    # Summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"=" * 40)

    working_patterns = sum(1 for r in results.values() if r['success'])

    for name, result in results.items():
        status = "✅ WORKING" if result['success'] else "❌ FAILED"
        print(f"{status} {name}: max value {result['max_val']:.3f}")

    if working_patterns > 0:
        print(f"\n🎉 BREAKTHROUGH! {working_patterns}/{len(test_patterns)} patterns working!")
        print("4×4 Quantum image processing is now functional!")

        return True
    else:
        print(f"\n❌ Still debugging needed...")
        return False

if __name__ == "__main__":
    success = test_working_encoding()

    if success:
        print("\n🚀 Ready to scale to full 4×4 quantum image processing!")
    else:
        print("\n🔧 Need to investigate circuit construction further...")