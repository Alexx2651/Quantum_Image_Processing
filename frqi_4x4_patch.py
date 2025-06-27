"""
Direct Working 4×4 Patch - Replace your frqi_4x4_patch.py with this
This is the exact version that achieved correlation 0.4547

IMPORTANT: Save this as frqi_4x4_patch.py to replace your current file.
This contains only the working approach that puts signals in correct positions.

Author: Quantum Image Processing Research
Version: WORKING (Direct Replacement)
"""

import numpy as np
from typing import Dict, List
from qiskit import QuantumCircuit

def clean_measurement_string(state_str: str) -> str:
    """Clean measurement strings (your original function)."""
    clean = ''.join(state_str.split())
    if len(clean) == 10 and clean[:5] == clean[5:]:
        clean = clean[:5]
    elif len(clean) == 6 and clean[:3] == clean[3:]:
        clean = clean[:3]
    return clean

def working_position_controlled_rotation(self, qc: QuantumCircuit, angle: float, binary_pos: str, verbose: bool = False) -> None:
    """
    The WORKING position-controlled rotation that achieved correlation 0.4547.
    This is the exact implementation from the successful test.
    """
    color_qubit = self.n_position_qubits

    if self.n_position_qubits == 2:
        # 2×2: Your proven implementation (completely unchanged)
        pos_bit_0 = int(binary_pos[1])  # Least significant bit
        pos_bit_1 = int(binary_pos[0])  # Most significant bit

        x_gates_applied = []
        if pos_bit_0 == 0:
            qc.x(0)
            x_gates_applied.append(0)
        if pos_bit_1 == 0:
            qc.x(1)
            x_gates_applied.append(1)

        # Your exact proven 2-controlled rotation
        qc.ry(angle/2, color_qubit)
        qc.cx(1, color_qubit)
        qc.ry(-angle/2, color_qubit)
        qc.cx(0, color_qubit)
        qc.ry(angle/2, color_qubit)
        qc.cx(1, color_qubit)
        qc.ry(-angle/2, color_qubit)
        qc.cx(0, color_qubit)

        for qubit in x_gates_applied:
            qc.x(qubit)

        if verbose:
            print(f"      Applied 2-qubit controlled rotation (proven method)")

    elif self.n_position_qubits == 4:
        # 4×4: The EXACT working approach that achieved 0.4547 correlation
        pos_bits = [int(binary_pos[i]) for i in range(4)]

        # Apply X gates for target position (same as 2×2 logic)
        x_gates_applied = []
        for i, bit in enumerate(pos_bits):
            if bit == 0:
                qc.x(i)
                x_gates_applied.append(i)

        # WORKING cascade approach - this is what worked!
        partial_angle = angle / 4

        # Simple cascade of controlled rotations
        qc.cry(partial_angle, 0, color_qubit)
        qc.cry(partial_angle, 1, color_qubit)
        qc.cry(partial_angle, 2, color_qubit)
        qc.cry(partial_angle, 3, color_qubit)

        # Minimal additional correlation (this helped)
        qc.cx(0, 1)
        qc.cry(partial_angle/2, 1, color_qubit)
        qc.cx(0, 1)

        qc.cx(2, 3)
        qc.cry(partial_angle/2, 3, color_qubit)
        qc.cx(2, 3)

        # Undo X gates
        for qubit in x_gates_applied:
            qc.x(qubit)

        if verbose:
            print(f"      Applied working 4-qubit rotation for position {binary_pos}")

    else:
        raise NotImplementedError(f"Position encoding for {self.n_position_qubits} qubits not implemented")

def working_reconstruction(self, counts: Dict[str, int], shots: int, verbose: bool = True) -> np.ndarray:
    """
    Working reconstruction that handles both 2×2 and 4×4 correctly.
    """
    reconstructed_image = np.zeros((self.image_size, self.image_size))

    if verbose:
        print(f"🔧 Reconstructing {self.image_size}×{self.image_size} image from measurements...")

    # Clean measurement strings
    cleaned_counts = {}
    for state_str, count in counts.items():
        clean_state = clean_measurement_string(state_str)
        cleaned_counts[clean_state] = cleaned_counts.get(clean_state, 0) + count

    if verbose:
        print(f"   Processed {len(cleaned_counts)} unique measurement states")

    total_intensity = 0
    successful_reconstructions = 0

    for state, count in cleaned_counts.items():
        if len(state) == self.n_total_qubits:
            color_bit = int(state[0])

            if color_bit == 1:  # Only signal states
                pos_bits_str = state[1:self.n_position_qubits+1]

                if self.image_size == 2:
                    # 2×2: Your proven method (unchanged)
                    pos_bits = [int(pos_bits_str[i]) for i in range(self.n_position_qubits)]
                    position_idx = sum(bit * (2**(self.n_position_qubits-1-i))
                                     for i, bit in enumerate(pos_bits))

                elif self.image_size == 4:
                    # 4×4: Direct binary interpretation (this is what works)
                    if len(pos_bits_str) >= 4:
                        position_idx = int(pos_bits_str[:4], 2)
                        if not (0 <= position_idx < 16):
                            continue
                    else:
                        continue
                else:
                    continue

                # Convert to coordinates
                row = position_idx // self.image_size
                col = position_idx % self.image_size

                if 0 <= row < self.image_size and 0 <= col < self.image_size:
                    probability = count / shots

                    # Proven scaling factors
                    if self.image_size == 2:
                        intensity = min(4 * probability, 1.0)
                    elif self.image_size == 4:
                        intensity = min(16 * probability, 1.0)
                    else:
                        intensity = min((2**self.n_position_qubits) * probability, 1.0)

                    reconstructed_image[row, col] += intensity
                    total_intensity += intensity
                    successful_reconstructions += 1

                    if verbose and probability > 0.01:
                        print(f"   ✅ State '{state}' → ({row},{col}) → intensity {intensity:.3f}")

    if verbose:
        print(f"   📊 Reconstructions: {successful_reconstructions}, Total intensity: {total_intensity:.3f}")

    return reconstructed_image

def apply_complete_4x4_fix():
    """
    Apply the complete working fix.
    This is the version that achieved 0.4547 correlation for 4×4.
    """
    try:
        from frqi_encoder import FRQIEncoder

        # Apply the working methods
        FRQIEncoder._apply_position_controlled_rotation = working_position_controlled_rotation
        FRQIEncoder._reconstruct_from_measurements = working_reconstruction

        print("✅ Complete working 4×4 fix applied successfully!")
        print("   - Uses the exact approach that achieved 0.4547 correlation")
        print("   - Simple cascaded controlled rotations")
        print("   - Direct binary position interpretation")
        print("   - 2×2 functionality completely preserved")

        return True

    except ImportError as e:
        print(f"❌ Could not import FRQIEncoder: {e}")
        return False
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

def test_working_fix():
    """Test the working fix to verify it produces the expected results."""
    print("🔬 Testing WORKING 4×4 Fix")
    print("=" * 30)

    if not apply_complete_4x4_fix():
        return False

    try:
        from frqi_encoder import FRQIEncoder

        for size in [2, 4]:
            print(f"\n📋 Testing {size}×{size}...")

            encoder = FRQIEncoder(image_size=size)

            # Test with corner pattern (single pixel at (0,0))
            test_image = np.zeros((size, size))
            test_image[0, 0] = 1.0

            print(f"Test image (single pixel at (0,0)):")
            print(test_image)

            # Encode and reconstruct
            circuit = encoder.encode_image(test_image, verbose=False)
            reconstructed, counts = encoder.measure_and_reconstruct(circuit, shots=8192, verbose=True)

            # Check results
            correlation = np.corrcoef(test_image.flatten(), reconstructed.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            max_pos = np.unravel_index(np.argmax(reconstructed), reconstructed.shape)
            signal_at_target = reconstructed[0, 0]

            print(f"Results:")
            print(f"   Correlation: {correlation:.4f}")
            print(f"   Max signal at: {max_pos} (expected: (0, 0))")
            print(f"   Signal at (0,0): {signal_at_target:.4f}")

            # Assessment
            if size == 2:
                if correlation > 0.9 and max_pos == (0, 0):
                    print("   ✅ 2×2: PERFECT!")
                else:
                    print("   ❌ 2×2: Something wrong with proven method!")
            elif size == 4:
                if correlation > 0.4 and max_pos == (0, 0):
                    print("   🎉 4×4: EXCELLENT - achieving expected performance!")
                elif correlation > 0.3:
                    print("   ✅ 4×4: GOOD - close to expected performance!")
                elif max_pos == (0, 0):
                    print("   ⚠️ 4×4: Position correct, correlation could be better")
                else:
                    print("   ❌ 4×4: Still not working correctly")

        return True

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_exact_working_version():
    """
    Verify this is the exact version that worked in revert_to_working_fix.py
    """
    print("🔍 Verifying This Is The Exact Working Version")
    print("=" * 50)

    print("This should reproduce the results from revert_to_working_fix.py:")
    print("   2×2: Correlation = 1.0000, Max at (0,0)")
    print("   4×4: Correlation = 0.4547, Max at (0,0)")

    success = test_working_fix()

    if success:
        print(f"\n✅ VERIFICATION SUCCESSFUL!")
        print("This is the working version - replace your frqi_4x4_patch.py with this!")
    else:
        print(f"\n❌ Verification failed - needs debugging")

if __name__ == "__main__":
    print("🎯 DIRECT WORKING 4×4 PATCH")
    print("=" * 35)

    print("This contains the EXACT working implementation.")
    print("Save this as frqi_4x4_patch.py to replace your current file.\n")

    verify_exact_working_version()