"""
Enhanced 8x8 Quantum Image Processor
Extends existing Quantum_Image_Processing repository for 8x8 image processing with IBM hardware support

This module integrates seamlessly with your existing:
- ScalableFRQIEncoder
- FullyFixedIBMQuantumTester
- OptimizedQuantumEdgeDetection

Author: Quantum Image Processing Research
Version: 1.0 (8x8 Extension)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
import time
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Import your existing modules
from scaled_quantum_processing import ScalableFRQIEncoder
from ibm_hardware_test import FullyFixedIBMQuantumTester
from integrated_quantum_edge_detection import OptimizedQuantumEdgeDetection


class Enhanced8x8QuantumProcessor:
    """
    Enhanced processor for 8x8 quantum image processing
    Integrates with existing repository infrastructure
    """

    def __init__(self):
        """Initialize the 8x8 processor"""
        self.image_size = 8
        self.n_pixels = 64
        self.n_position_qubits = 6  # log2(64)
        self.n_color_qubits = 1
        self.n_total_qubits = 7

        print(f"🚀 Enhanced 8x8 Quantum Processor Initialized")
        print(f"   Image size: 8×8 ({self.n_pixels} pixels)")
        print(f"   Total qubits required: {self.n_total_qubits}")
        print(f"   Compatible with existing ScalableFRQIEncoder")

        # Initialize core components
        try:
            self.encoder = ScalableFRQIEncoder(image_size=8)
            self.ibm_tester = None  # Initialize on demand
            self.edge_detector = None  # Initialize on demand
            print(f"✅ Core components ready")
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")

    def create_8x8_test_patterns(self) -> Dict[str, np.ndarray]:
        """Create comprehensive 8x8 test patterns"""
        patterns = {}

        # Start with simple patterns (lowest complexity)
        single = np.zeros((8, 8))
        single[3, 3] = 1.0
        patterns['single'] = single

        # Corner pixels (medium complexity)
        corners = np.zeros((8, 8))
        corners[0, 0] = 1.0  # Top-left
        corners[0, 7] = 1.0  # Top-right
        corners[7, 0] = 1.0  # Bottom-left
        corners[7, 7] = 1.0  # Bottom-right
        patterns['corners'] = corners

        # Cross pattern (higher complexity)
        cross = np.zeros((8, 8))
        cross[3:5, :] = 1.0  # Horizontal bar
        cross[:, 3:5] = 1.0  # Vertical bar
        patterns['cross'] = cross

        # Edge pattern (highest complexity)
        edge = np.zeros((8, 8))
        edge[0, :] = 1.0  # Top edge
        edge[7, :] = 1.0  # Bottom edge
        edge[:, 0] = 1.0  # Left edge
        edge[:, 7] = 1.0  # Right edge
        patterns['edge'] = edge

        # Checkerboard pattern (very high complexity)
        checkerboard = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    checkerboard[i, j] = 1.0
        patterns['checkerboard'] = checkerboard

        return patterns

    def estimate_complexity(self, image: np.ndarray) -> Dict:
        """Estimate circuit complexity for the image"""
        non_zero_pixels = np.count_nonzero(image)

        # Based on your FRQI implementation patterns
        estimated_rotation_gates = non_zero_pixels
        estimated_cnot_gates = non_zero_pixels * (self.n_position_qubits - 1)
        total_gates = estimated_rotation_gates + estimated_cnot_gates

        complexity_level = "LOW" if non_zero_pixels < 8 else \
            "MEDIUM" if non_zero_pixels < 20 else \
                "HIGH" if non_zero_pixels < 40 else "VERY HIGH"

        return {
            'non_zero_pixels': non_zero_pixels,
            'total_pixels': self.n_pixels,
            'sparsity': 1 - (non_zero_pixels / self.n_pixels),
            'estimated_gates': total_gates,
            'complexity_level': complexity_level,
            'hardware_recommended': non_zero_pixels < 32  # Conservative for NISQ
        }

    def test_8x8_pattern(self, pattern_name: str, image: np.ndarray,
                         test_simulation: bool = True, test_hardware: bool = False,
                         shots: int = 4096) -> Dict:
        """
        Comprehensive test of an 8x8 pattern using existing infrastructure
        """
        print(f"\n🔬 Testing 8x8 {pattern_name.upper()} Pattern")
        print("=" * 55)

        # Show pattern info
        complexity = self.estimate_complexity(image)
        print(f"Pattern: {image.shape} with {complexity['non_zero_pixels']} non-zero pixels")
        print(f"Complexity: {complexity['complexity_level']}")
        print(f"Hardware recommended: {'✅' if complexity['hardware_recommended'] else '⚠️'}")

        results = {
            'pattern_name': pattern_name,
            'image': image,
            'complexity': complexity,
            'simulation_success': False,
            'hardware_success': False
        }

        # Simulation test using your ScalableFRQIEncoder
        if test_simulation:
            print(f"\n📊 Simulation Test (shots={shots})")
            try:
                # Encode using your existing encoder
                circuit = self.encoder.encode_image(image, verbose=True)

                print(f"   Circuit stats: {circuit.num_qubits} qubits, {circuit.depth()} depth")
                print(f"   Gate counts: {circuit.count_ops()}")

                # Reconstruct using your existing methods
                reconstructed, counts, exec_time = self.encoder.measure_and_reconstruct(
                    circuit, shots=shots, verbose=True
                )

                # Calculate metrics
                correlation = self._calculate_correlation(image, reconstructed)
                mse = np.mean((image - reconstructed) ** 2)

                results.update({
                    'simulation_success': True,
                    'sim_correlation': correlation,
                    'sim_mse': mse,
                    'sim_execution_time': exec_time,
                    'sim_reconstructed': reconstructed,
                    'circuit_depth': circuit.depth(),
                    'gate_count': sum(circuit.count_ops().values())
                })

                print(f"   ✅ Simulation results:")
                print(f"      Correlation: {correlation:.4f}")
                print(f"      MSE: {mse:.6f}")
                print(f"      Execution time: {exec_time:.3f}s")

                # Visualize results
                self._visualize_results(image, reconstructed, pattern_name, correlation, "Simulation")

            except Exception as e:
                print(f"   ❌ Simulation failed: {e}")
                results['simulation_error'] = str(e)

        # Hardware test using your IBM infrastructure
        if test_hardware and results.get('simulation_success', False):
            print(f"\n🚀 IBM Hardware Test")
            try:
                # Initialize IBM tester if not done
                if self.ibm_tester is None:
                    self.ibm_tester = FullyFixedIBMQuantumTester()

                if self.ibm_tester.service:
                    print("✅ Connected to IBM Quantum")

                    # Use reduced shots for hardware to manage cost
                    hw_shots = min(shots, 1024)

                    # Test using your existing framework
                    hw_results = self.ibm_tester.test_optimized_frqi_fixed(
                        pattern=pattern_name,
                        image_size=8,
                        shots=hw_shots
                    )

                    if hw_results:
                        ideal_corr = hw_results.get('ideal_correlation', 0)
                        hw_corr = hw_results.get('hardware_correlation', 0)

                        results.update({
                            'hardware_success': True,
                            'hw_ideal_correlation': ideal_corr,
                            'hw_hardware_correlation': hw_corr,
                            'hw_shots': hw_shots
                        })

                        print(f"   📊 Hardware results:")
                        print(f"      Ideal correlation: {ideal_corr:.4f}")
                        print(f"      Hardware correlation: {hw_corr:.4f}")

                        if hw_corr > 0.3:
                            print(f"   ✅ Hardware test successful!")
                        else:
                            print(f"   ⚠️ Hardware correlation low (NISQ noise effects)")
                    else:
                        print(f"   ⚠️ Hardware test completed but needs analysis")

                else:
                    print("❌ IBM Quantum not available - check configuration")

            except Exception as e:
                print(f"   ❌ Hardware test failed: {e}")
                results['hardware_error'] = str(e)

        return results

    def test_8x8_edge_detection(self, pattern_name: str, image: np.ndarray) -> Dict:
        """Test 8x8 quantum edge detection using your existing framework"""
        print(f"\n🔍 8x8 Quantum Edge Detection Test: {pattern_name.upper()}")
        print("=" * 50)

        try:
            # Initialize edge detector if not done
            if self.edge_detector is None:
                self.edge_detector = OptimizedQuantumEdgeDetection(image_size=8)

            # Run comprehensive analysis using your existing method
            edge_results = self.edge_detector.run_optimized_analysis(pattern_name, shots=4096)

            if edge_results.get('frqi_success', False):
                frqi_corr = edge_results.get('frqi_correlation', 0)
                print(f"✅ 8x8 FRQI reconstruction: {frqi_corr:.4f} correlation")

                # Test edge detection if FRQI works
                if 'edge_success' in edge_results and edge_results['edge_success']:
                    edge_corr = edge_results.get('edge_correlation', 0)
                    print(f"✅ 8x8 Edge detection: {edge_corr:.4f} correlation")
                else:
                    print(f"⚠️ Edge detection needs optimization for 8x8")
            else:
                print(f"❌ 8x8 FRQI reconstruction failed")

            return edge_results

        except Exception as e:
            print(f"❌ 8x8 Edge detection failed: {e}")
            return {'error': str(e)}

    def run_comprehensive_8x8_test(self, test_hardware: bool = False):
        """Run comprehensive 8x8 testing suite"""
        print(f"\n🎯 COMPREHENSIVE 8x8 QUANTUM IMAGE PROCESSING TEST")
        print("=" * 70)
        print(f"Hardware testing: {'✅ ENABLED' if test_hardware else '❌ DISABLED'}")

        # Create test patterns
        patterns = self.create_8x8_test_patterns()

        all_results = {}

        # Test patterns in order of increasing complexity
        test_order = ['single', 'corners', 'cross', 'edge', 'checkerboard']

        for pattern_name in test_order:
            if pattern_name not in patterns:
                continue

            image = patterns[pattern_name]

            print(f"\nPattern: {pattern_name}")
            print(image)

            # Test the pattern
            results = self.test_8x8_pattern(
                pattern_name, image,
                test_simulation=True,
                test_hardware=test_hardware,
                shots=4096
            )

            all_results[pattern_name] = results

            # Stop if simulation fails (indicates fundamental issue)
            if not results.get('simulation_success', False):
                print(f"⚠️ Stopping tests due to simulation failure")
                break

        # Summary report
        self._generate_summary_report(all_results)

        return all_results

    def _calculate_correlation(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate correlation between original and reconstructed images"""
        try:
            flat_orig = original.flatten()
            flat_recon = reconstructed.flatten()

            if np.var(flat_orig) < 1e-10 and np.var(flat_recon) < 1e-10:
                return 1.0 if np.allclose(flat_orig, flat_recon, atol=1e-10) else 0.0
            elif np.var(flat_orig) < 1e-10 or np.var(flat_recon) < 1e-10:
                return 0.0
            else:
                correlation = np.corrcoef(flat_orig, flat_recon)[0, 1]
                return 0.0 if np.isnan(correlation) else correlation
        except:
            return 0.0

    def _visualize_results(self, original: np.ndarray, reconstructed: np.ndarray,
                           pattern_name: str, correlation: float, test_type: str):
        """Visualize 8x8 test results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        im1 = axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Original 8x8 {pattern_name}')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0])

        # Reconstructed image
        max_val = max(np.max(reconstructed), 1.0)
        im2 = axes[1].imshow(reconstructed, cmap='gray', vmin=0, vmax=max_val)
        axes[1].set_title(f'{test_type} Reconstruction\nCorr: {correlation:.3f}')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[1])

        # Error visualization
        error = np.abs(original - reconstructed)
        im3 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=np.max(error))
        axes[2].set_title(f'Absolute Error\nMSE: {np.mean(error ** 2):.6f}')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()

    def _generate_summary_report(self, all_results: Dict):
        """Generate comprehensive summary report"""
        print(f"\n📊 8x8 QUANTUM IMAGE PROCESSING SUMMARY REPORT")
        print("=" * 60)

        successful_sims = 0
        successful_hw = 0
        total_patterns = len(all_results)

        print(f"{'Pattern':<12} {'Sim':<5} {'HW':<4} {'Sim Corr':<10} {'HW Corr':<9} {'Gates':<7}")
        print("-" * 60)

        for pattern, results in all_results.items():
            sim_success = "✅" if results.get('simulation_success') else "❌"
            hw_success = "✅" if results.get('hardware_success') else "❌"

            sim_corr = results.get('sim_correlation', 0)
            hw_corr = results.get('hw_hardware_correlation', 0)
            gates = results.get('gate_count', 0)

            print(f"{pattern:<12} {sim_success:<5} {hw_success:<4} {sim_corr:<10.3f} {hw_corr:<9.3f} {gates:<7}")

            if results.get('simulation_success'):
                successful_sims += 1
            if results.get('hardware_success'):
                successful_hw += 1

        print("-" * 60)
        print(
            f"Success rate - Simulation: {successful_sims}/{total_patterns}, Hardware: {successful_hw}/{total_patterns}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if successful_sims == total_patterns:
            print("✅ All simulation tests passed - 8x8 scaling successful!")
        elif successful_sims > 0:
            print("⚠️ Some patterns work - optimize complex patterns")
        else:
            print("❌ Simulation issues - check basic FRQI implementation")

        if successful_hw > 0:
            print("✅ IBM hardware testing successful - ready for real quantum computing!")
        elif successful_sims > 0:
            print("💡 Try hardware testing with simpler patterns first")


def main():
    """Main test function for 8x8 quantum image processing"""
    print("🚀 STARTING 8x8 QUANTUM IMAGE PROCESSING INTEGRATION TEST")

    processor = Enhanced8x8QuantumProcessor()

    # Quick test
    print("\n--- Quick 8x8 Test ---")
    patterns = processor.create_8x8_test_patterns()

    # Test simplest pattern first
    quick_result = processor.test_8x8_pattern(
        "single", patterns["single"],
        test_simulation=True, test_hardware=False
    )

    if quick_result.get('simulation_success'):
        print("\n✅ 8x8 basic functionality confirmed!")

        # Ask user for comprehensive test
        user_input = input("\nRun comprehensive test suite? (y/n): ").strip().lower()
        if user_input == 'y':
            # Run comprehensive test
            hw_test = input("Include IBM hardware testing? (y/n): ").strip().lower() == 'y'
            processor.run_comprehensive_8x8_test(test_hardware=hw_test)
        else:
            print("Quick test complete. Run main() again for full suite.")
    else:
        print("❌ 8x8 basic test failed - check configuration")


if __name__ == "__main__":
    main()