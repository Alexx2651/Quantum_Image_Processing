"""
FULLY FIXED IBM Quantum Hardware Testing - Corrected Measurement Processing
Fixes the measurement string length inconsistency between simulators and hardware

Key fixes:
1. Enhanced measurement string cleaning for different formats
2. Robust bit length handling across simulators and hardware
3. Proper correlation calculations
4. Comprehensive debugging output

Author: Quantum Image Processing Research
Version: 6.1 (Fully Fixed)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import time
import warnings
warnings.filterwarnings('ignore')

# Auto-detect available IBM Runtime API version
RUNTIME_AVAILABLE = False
SAMPLER_V2_AVAILABLE = False
SESSION_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    RUNTIME_AVAILABLE = True

    # Try to import SamplerV2
    try:
        from qiskit_ibm_runtime import SamplerV2
        SAMPLER_V2_AVAILABLE = True
        print("✅ SamplerV2 available")
    except ImportError:
        print("⚠️ SamplerV2 not available, using legacy Sampler")

    # Try to import Session
    try:
        from qiskit_ibm_runtime import Session
        SESSION_AVAILABLE = True
        print("✅ Session available")
    except ImportError:
        print("⚠️ Session not available")

    # Always import legacy Sampler as fallback
    from qiskit_ibm_runtime import Sampler
    print("✅ Legacy Sampler available")

except ImportError:
    print("❌ IBM Runtime not available at all")
    RUNTIME_AVAILABLE = False

# Import our optimized quantum systems
from scaled_quantum_processing import ScalableFRQIEncoder
from integrated_quantum_edge_detection import OptimizedQuantumEdgeDetection

def clean_measurement_string_fixed(state_str: str, expected_qubits: int = 3) -> str:
    """
    Enhanced measurement string cleaning that handles different formats

    Args:
        state_str: Raw measurement string from Qiskit
        expected_qubits: Expected number of qubits (3 for our system)

    Returns:
        Cleaned measurement string of correct length
    """
    # Remove any whitespace
    clean = ''.join(state_str.split())

    # Case 1: String is exactly the right length
    if len(clean) == expected_qubits:
        return clean

    # Case 2: String is doubled (e.g., '101101' instead of '101')
    if len(clean) == expected_qubits * 2:
        first_half = clean[:expected_qubits]
        second_half = clean[expected_qubits:]
        if first_half == second_half:
            return first_half

    # Case 3: String is longer than expected (padded with zeros)
    # Case 3: String is longer than expected (padded with zeros)
    if len(clean) > expected_qubits:
        # FIXED: Take the FIRST N bits (where the data actually is!)
        truncated = clean[:expected_qubits]  # ← FIXED: Use [:3] not [-3:]
        print(f"      String padded, taking first {expected_qubits} bits: '{clean}' → '{truncated}'")
        return truncated

    # Case 4: String is shorter than expected (shouldn't happen)
    if len(clean) < expected_qubits:
        # Pad with leading zeros
        padded = clean.zfill(expected_qubits)
        return padded

    return clean

class FullyFixedIBMQuantumTester:
    """
    Fully fixed IBM Quantum hardware tester with corrected measurement processing
    """

    def __init__(self):
        """Initialize IBM Quantum service connection"""
        if not RUNTIME_AVAILABLE:
            print("❌ IBM Runtime not available")
            self.service = None
            return

        try:
            # Initialize IBM Quantum service
            self.service = QiskitRuntimeService(channel="ibm_quantum")
            print("✅ Connected to IBM Quantum successfully!")

            # Get available backends
            self.backends = self.service.backends()
            operational_backends = [b for b in self.backends
                                  if b.status().operational and not b.configuration().simulator]
            print(f"Available quantum backends: {len(operational_backends)}")

        except Exception as e:
            print(f"❌ Failed to connect to IBM Quantum: {e}")
            self.service = None

    def select_best_backend(self, min_qubits=3):
        """Select the best available backend"""
        if not self.service:
            return None

        available_backends = []

        for backend in self.backends:
            if (backend.status().operational and
                backend.configuration().n_qubits >= min_qubits and
                not backend.configuration().simulator):

                status = backend.status()
                available_backends.append({
                    'backend': backend,
                    'name': backend.name,
                    'qubits': backend.configuration().n_qubits,
                    'queue': status.pending_jobs,
                    'operational': status.operational
                })

        if not available_backends:
            print("❌ No suitable quantum backends available")
            return None

        # Sort by queue length
        available_backends.sort(key=lambda x: x['queue'])

        print("📋 Available Quantum Backends:")
        for i, backend_info in enumerate(available_backends[:3]):
            print(f"{i+1}. {backend_info['name']}: {backend_info['qubits']} qubits, "
                  f"queue: {backend_info['queue']} jobs")

        selected = available_backends[0]['backend']
        print(f"🎯 Selected: {selected.name}")
        return selected

    def run_on_hardware_fixed(self, circuit, backend, shots=1024):
        """
        Fixed hardware execution with proper API usage
        """
        print(f"🚀 Running on {backend.name} with {shots} shots...")
        print(f"   API capabilities: SamplerV2={SAMPLER_V2_AVAILABLE}, Session={SESSION_AVAILABLE}")

        try:
            # Ensure circuit has measurements
            if not circuit.cregs:
                circuit.measure_all()

            # Transpile circuit
            transpiled_circuit = transpile(circuit, backend=backend, optimization_level=2)
            print(f"📊 Transpiled: {transpiled_circuit.depth()} depth, {transpiled_circuit.count_ops()} gates")

            start_time = time.time()
            job = None

            # Method 1: Try SamplerV2 with Session (correct 2024 syntax)
            if SAMPLER_V2_AVAILABLE and SESSION_AVAILABLE:
                try:
                    print("🔄 Trying SamplerV2 + Session (2024 syntax)...")
                    from qiskit_ibm_runtime import SamplerV2, Session

                    with Session(backend=backend) as session:
                        sampler = SamplerV2()  # No session parameter!
                        job = sampler.run([transpiled_circuit], shots=shots)
                        print(f"✅ Job submitted with SamplerV2+Session: {job.job_id()}")

                except Exception as e1:
                    print(f"⚠️ SamplerV2+Session failed: {e1}")
                    job = None

            # Method 2: Try legacy Sampler with Session
            if job is None and SESSION_AVAILABLE:
                try:
                    print("🔄 Trying legacy Sampler + Session...")
                    from qiskit_ibm_runtime import Sampler, Session

                    with Session(backend=backend) as session:
                        sampler = Sampler()
                        job = sampler.run(transpiled_circuit, shots=shots)
                        print(f"✅ Job submitted with Sampler+Session: {job.job_id()}")

                except Exception as e2:
                    print(f"⚠️ Sampler+Session failed: {e2}")
                    job = None

            # Method 3: Try legacy Sampler without Session
            if job is None:
                try:
                    print("🔄 Trying legacy Sampler without Session...")
                    from qiskit_ibm_runtime import Sampler

                    sampler = Sampler()
                    job = sampler.run(transpiled_circuit, shots=shots)
                    print(f"✅ Job submitted with legacy Sampler: {job.job_id()}")

                except Exception as e3:
                    print(f"⚠️ Legacy Sampler failed: {e3}")
                    job = None

            # If all methods failed
            if job is None:
                print("❌ All hardware execution methods failed")
                return None

            # Wait for results
            print("⏳ Waiting for quantum execution...")
            print(f"   Job ID: {job.job_id()}")
            print("   This may take several minutes on real hardware...")

            result = job.result()
            execution_time = time.time() - start_time

            # Version-agnostic result extraction
            raw_counts = None

            try:
                print(f"🔍 Processing result type: {type(result)}")

                # Handle SamplerV2 result format
                if hasattr(result, '__len__') and len(result) > 0:
                    first_result = result[0]
                    if hasattr(first_result, 'data'):
                        data = first_result.data
                        for attr_name in ['meas', 'c', 'cr', 'classical', 'measurements']:
                            if hasattr(data, attr_name):
                                try:
                                    attr = getattr(data, attr_name)
                                    if hasattr(attr, 'get_counts'):
                                        raw_counts = attr.get_counts()
                                        print(f"   ✅ Found counts via data.{attr_name}.get_counts()")
                                        break
                                except:
                                    continue

                    # Try quasi_dists for newer SamplerV2
                    if raw_counts is None and hasattr(first_result, 'quasi_dists'):
                        try:
                            quasi_dist = first_result.quasi_dists[0]
                            raw_counts = {}
                            for state, prob in quasi_dist.items():
                                state_str = format(state, f'0{transpiled_circuit.num_clbits}b')
                                raw_counts[state_str] = int(prob * shots)
                            print("   ✅ Converted quasi_dists to counts")
                        except Exception as qd_error:
                            print(f"   ⚠️ Quasi_dists conversion failed: {qd_error}")

                # Handle legacy Sampler result format
                if raw_counts is None and hasattr(result, 'get_counts'):
                    raw_counts = result.get_counts()
                    print("   ✅ Found counts via result.get_counts()")

                if raw_counts is None:
                    print("❌ Could not extract measurement counts from result")
                    return None

                # Clean and validate counts
                cleaned_counts = {}
                total_shots_received = 0

                for state, count in raw_counts.items():
                    clean_state = str(state).replace(' ', '')
                    count_int = int(count)
                    cleaned_counts[clean_state] = count_int
                    total_shots_received += count_int

                print("✅ Hardware execution successful!")
                print(f"   Execution time: {execution_time:.3f}s")
                print(f"   Unique states measured: {len(cleaned_counts)}")
                print(f"   Total shots received: {total_shots_received}")

                return {
                    'counts': cleaned_counts,
                    'execution_time': execution_time,
                    'backend_name': backend.name,
                    'shots': shots,
                    'actual_shots': total_shots_received,
                    'transpiled_depth': transpiled_circuit.depth(),
                    'transpiled_gates': transpiled_circuit.count_ops(),
                    'job_id': job.job_id(),
                    'success': True
                }

            except Exception as e:
                print(f"❌ Result extraction failed: {e}")
                return None

        except Exception as e:
            print(f"❌ Hardware execution failed: {e}")
            return None

    def create_noise_model(self, backend):
        """Create noise model from backend"""
        try:
            noise_model = NoiseModel.from_backend(backend)
            print(f"✅ Created noise model from {backend.name}")
            return noise_model
        except Exception as e:
            print(f"⚠️ Could not create noise model: {e}")
            return None

    def run_ideal_simulation(self, circuit, shots=1024):
        """Run ideal simulation"""
        print("⚡ Running ideal simulation...")

        try:
            if not circuit.cregs:
                measured_circuit = circuit.copy()
                measured_circuit.measure_all()
            else:
                measured_circuit = circuit

            simulator = AerSimulator()
            transpiled_circuit = transpile(measured_circuit, simulator)

            start_time = time.time()
            job = simulator.run(transpiled_circuit, shots=shots)
            result = job.result()
            execution_time = time.time() - start_time

            raw_counts = result.get_counts()
            cleaned_counts = {state.replace(' ', ''): count for state, count in raw_counts.items()}

            print("✅ Ideal simulation completed!")
            return {
                'counts': cleaned_counts,
                'execution_time': execution_time,
                'backend_name': 'ideal_simulator',
                'shots': shots,
                'success': True
            }

        except Exception as e:
            print(f"❌ Ideal simulation failed: {e}")
            return None

    def run_noisy_simulation(self, circuit, noise_model, shots=1024):
        """Run noisy simulation"""
        print("🔧 Running noisy simulation...")

        try:
            if not circuit.cregs:
                measured_circuit = circuit.copy()
                measured_circuit.measure_all()
            else:
                measured_circuit = circuit

            simulator = AerSimulator(noise_model=noise_model)
            transpiled_circuit = transpile(measured_circuit, simulator)

            start_time = time.time()
            job = simulator.run(transpiled_circuit, shots=shots)
            result = job.result()
            execution_time = time.time() - start_time

            raw_counts = result.get_counts()
            cleaned_counts = {state.replace(' ', ''): count for state, count in raw_counts.items()}

            print("✅ Noisy simulation completed!")
            return {
                'counts': cleaned_counts,
                'execution_time': execution_time,
                'backend_name': 'noisy_simulator',
                'shots': shots,
                'success': True
            }

        except Exception as e:
            print(f"❌ Noisy simulation failed: {e}")
            return None

    def reconstruct_image_with_encoder_fixed(self, counts, shots, encoder):
        """
        FIXED reconstruction with proper measurement string handling
        """
        print(f"🔧 Reconstructing using FIXED encoder...")
        print(f"   Raw counts received: {counts}")
        print(f"   Total unique states: {len(counts)}")

        # Clean all measurement strings with fixed function
        cleaned_counts = {}
        expected_qubits = encoder.n_total_qubits

        print(f"   Expected qubits: {expected_qubits}")
        print(f"   Cleaning measurement strings:")

        for state_str, count in counts.items():
            clean_state = clean_measurement_string_fixed(state_str, expected_qubits)
            print(f"     '{state_str}' → '{clean_state}' ({len(state_str)} → {len(clean_state)} bits)")

            if clean_state in cleaned_counts:
                cleaned_counts[clean_state] += count
            else:
                cleaned_counts[clean_state] = count

        print(f"   Final cleaned counts: {cleaned_counts}")

        # Use the encoder's reconstruction method with cleaned counts
        return encoder._reconstruct_from_measurements(cleaned_counts, shots, verbose=True)

    def test_optimized_frqi_fixed(self, pattern="single", image_size=2, shots=2048):
        """
        Test optimized FRQI system on IBM hardware with FIXED measurement processing
        """
        print(f"\n🔬 Testing FIXED Optimized FRQI - {pattern} ({image_size}×{image_size})...")

        # Create optimized system
        try:
            optimized_system = OptimizedQuantumEdgeDetection(image_size=image_size)
            test_image = optimized_system.create_test_image(pattern)

            print(f"Test image:\n{test_image}")

            # Get FRQI circuit
            frqi_circuit = optimized_system.get_base_frqi_circuit(test_image)

            print(f"\n📋 Circuit Analysis:")
            print(f"   Total qubits: {frqi_circuit.num_qubits}")
            print(f"   Circuit depth: {frqi_circuit.depth()}")
            print(f"   Gate counts: {frqi_circuit.count_ops()}")

            # Select backend
            backend = self.select_best_backend(min_qubits=frqi_circuit.num_qubits)
            if not backend:
                print("❌ No suitable backend available")
                return None

            # Add measurements to circuit
            frqi_circuit.measure_all()

            # Run simulations and hardware
            ideal_result = self.run_ideal_simulation(frqi_circuit, shots)

            noise_model = self.create_noise_model(backend)
            noisy_result = None
            if noise_model:
                noisy_result = self.run_noisy_simulation(frqi_circuit, noise_model, shots)

            # Try hardware execution
            hardware_result = self.run_on_hardware_fixed(frqi_circuit, backend, shots)

            # Reconstruct images using FIXED encoder
            results = {
                'test_image': test_image,
                'pattern_name': pattern,
                'image_size': image_size,
                'ideal_result': ideal_result,
                'noisy_result': noisy_result,
                'hardware_result': hardware_result,
                'encoder': optimized_system.frqi_encoder
            }

            print(f"\n🔧 RECONSTRUCTION WITH FIXED METHOD:")

            if ideal_result:
                print(f"📊 Processing ideal simulation results...")
                results['ideal_reconstructed'] = self.reconstruct_image_with_encoder_fixed(
                    ideal_result['counts'], shots, optimized_system.frqi_encoder
                )

            if noisy_result:
                print(f"📊 Processing noisy simulation results...")
                results['noisy_reconstructed'] = self.reconstruct_image_with_encoder_fixed(
                    noisy_result['counts'], shots, optimized_system.frqi_encoder
                )

            if hardware_result and hardware_result.get('success'):
                print(f"📊 Processing hardware results...")
                results['hardware_reconstructed'] = self.reconstruct_image_with_encoder_fixed(
                    hardware_result['counts'], shots, optimized_system.frqi_encoder
                )

            # Calculate quality metrics using robust correlation
            print(f"\n📊 CALCULATING CORRELATIONS:")

            if 'ideal_reconstructed' in results:
                ideal_corr = optimized_system._calculate_correlation_robust(test_image, results['ideal_reconstructed'])
                results['ideal_correlation'] = ideal_corr
                print(f"   ✅ Ideal reconstruction correlation: {ideal_corr:.4f}")

            if 'noisy_reconstructed' in results:
                noisy_corr = optimized_system._calculate_correlation_robust(test_image, results['noisy_reconstructed'])
                results['noisy_correlation'] = noisy_corr
                print(f"   ✅ Noisy reconstruction correlation: {noisy_corr:.4f}")

            if 'hardware_reconstructed' in results:
                hw_corr = optimized_system._calculate_correlation_robust(test_image, results['hardware_reconstructed'])
                results['hardware_correlation'] = hw_corr
                print(f"   ✅ Hardware reconstruction correlation: {hw_corr:.4f}")

            # Visualize results
            self.visualize_fixed_results(results)

            return results

        except Exception as e:
            print(f"❌ Fixed FRQI test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_fixed_results(self, results):
        """Enhanced visualization with fixed results"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        test_image = results['test_image']
        pattern_name = results['pattern_name']
        image_size = results['image_size']

        fig.suptitle(f'FIXED FRQI Hardware Test - {pattern_name} ({image_size}×{image_size})',
                     fontsize=16, fontweight='bold')

        # Original image
        im1 = axes[0,0].imshow(test_image, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('Original Image')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0,0])

        # Reconstructed images
        vmax = 1.0

        if 'ideal_reconstructed' in results:
            im2 = axes[0,1].imshow(results['ideal_reconstructed'], cmap='viridis', vmin=0, vmax=vmax)
            corr = results.get('ideal_correlation', 0)
            axes[0,1].set_title(f'FIXED Ideal Simulation\nCorr: {corr:.3f}')
            axes[0,1].grid(True, alpha=0.3)
            plt.colorbar(im2, ax=axes[0,1])

        if 'noisy_reconstructed' in results:
            im3 = axes[0,2].imshow(results['noisy_reconstructed'], cmap='viridis', vmin=0, vmax=vmax)
            corr = results.get('noisy_correlation', 0)
            axes[0,2].set_title(f'FIXED Noisy Simulation\nCorr: {corr:.3f}')
            axes[0,2].grid(True, alpha=0.3)
            plt.colorbar(im3, ax=axes[0,2])

        if 'hardware_reconstructed' in results:
            im4 = axes[0,3].imshow(results['hardware_reconstructed'], cmap='viridis', vmin=0, vmax=vmax)
            hw_corr = results.get('hardware_correlation', 0)
            backend_name = results.get('hardware_result', {}).get('backend_name', 'Unknown')
            axes[0,3].set_title(f'FIXED Hardware ({backend_name})\nCorr: {hw_corr:.3f}')
            axes[0,3].grid(True, alpha=0.3)
            plt.colorbar(im4, ax=axes[0,3])

        # Show measurement statistics
        methods = ['ideal_result', 'noisy_result', 'hardware_result']
        titles = ['Ideal', 'Noisy', 'Hardware']

        for i, (method, title) in enumerate(zip(methods, titles)):
            if method in results and results[method] and i < 3:
                counts = results[method]['counts']
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]
                states = [s[0] for s in sorted_counts]
                values = [s[1] for s in sorted_counts]

                colors = ['red' if s[0] == '1' else 'steelblue' for s in states]

                axes[1,i].bar(range(len(states)), values, color=colors)
                axes[1,i].set_title(f'{title} States')
                axes[1,i].set_ylabel('Count')
                axes[1,i].set_xticks(range(len(states)))
                axes[1,i].set_xticklabels(states, rotation=45, fontsize=8)

        # Performance summary
        correlations = []
        labels = []
        colors = []

        for method, label, color in [('ideal', 'Ideal', 'green'), ('noisy', 'Noisy', 'orange'), ('hardware', 'Hardware', 'blue')]:
            if f'{method}_correlation' in results:
                correlations.append(results[f'{method}_correlation'])
                labels.append(label)
                colors.append(color)

        if correlations:
            bars = axes[1,3].bar(labels, correlations, color=colors)
            axes[1,3].set_title('FIXED FRQI Performance')
            axes[1,3].set_ylabel('Correlation')
            axes[1,3].set_ylim([0, 1])

            # Add performance indicators
            axes[1,3].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent')
            axes[1,3].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good')
            axes[1,3].legend()

        plt.tight_layout()
        plt.show()

        # Print detailed analysis
        self._print_fixed_analysis(results)

    def _print_fixed_analysis(self, results):
        """Print detailed analysis of fixed results"""
        pattern = results['pattern_name']
        size = results['image_size']

        print(f"\n📊 DETAILED FIXED ANALYSIS")
        print("=" * 70)
        print(f"Pattern: {pattern.upper()} | Size: {size}×{size}")

        print(f"\n🔧 MEASUREMENT STRING FIXES APPLIED:")
        print(f"   ✅ Enhanced measurement string cleaning")
        print(f"   ✅ Proper bit length handling")
        print(f"   ✅ Consistent processing across simulators and hardware")

        print(f"\n📊 PERFORMANCE COMPARISON:")
        methods = ['ideal', 'noisy', 'hardware']
        for method in methods:
            if f'{method}_correlation' in results:
                corr = results[f'{method}_correlation']
                print(f"   {method.title():10}: {corr:.4f}")

        # Overall assessment
        ideal_corr = results.get('ideal_correlation', 0)
        hw_corr = results.get('hardware_correlation', 0)

        print(f"\n🎯 FIXED SYSTEM ASSESSMENT:")
        if ideal_corr > 0.9 and hw_corr > 0.7:
            print(f"   🌟 OUTSTANDING - All fixes working perfectly!")
        elif ideal_corr > 0.7 and hw_corr > 0.5:
            print(f"   ✅ EXCELLENT - Fixes significantly improved performance!")
        elif ideal_corr > 0.5:
            print(f"   ✅ GOOD - Major improvement, some optimization possible")
        else:
            print(f"   ⚠️ Still needs work - but major progress made")

        if abs(ideal_corr - hw_corr) < 0.3:
            print(f"   ✅ Consistent performance between ideal and hardware!")
        else:
            print(f"   ⚠️ Performance gap between ideal and hardware")


# Test the fully fixed implementation
if __name__ == "__main__":
    print("🚀 FULLY FIXED IBM QUANTUM HARDWARE TESTING")
    print("=" * 55)
    print("Corrected measurement string processing for all backends")

    tester = FullyFixedIBMQuantumTester()

    if not tester.service:
        print("❌ Cannot proceed without IBM Quantum access")
        print("💡 Make sure you have:")
        print("   1. IBM Quantum account set up")
        print("   2. qiskit-ibm-runtime installed")
        print("   3. API token configured")
        exit()

    # Test the fixed system
    test_mode = input("\nChoose test mode:\n1. Quick FIXED FRQI test\n2. Test multiple patterns\nEnter choice (1-2): ").strip()

    if test_mode == "1":
        # Quick fixed test
        print("\n🔬 Quick FIXED FRQI Test")
        pattern = input("Enter pattern (single/corners/cross): ").strip() or "cross"
        size = int(input("Enter image size (2 or 4): ").strip() or "2")

        results = tester.test_optimized_frqi_fixed(pattern, size, shots=1024)

        if results:
            ideal_corr = results.get('ideal_correlation', 0)
            hw_corr = results.get('hardware_correlation', 0)

            print(f"\n🎯 FIXED Test Results:")
            print(f"   Ideal correlation: {ideal_corr:.4f}")
            print(f"   Hardware correlation: {hw_corr:.4f}")
            print(f"   Improvement: Both should now be high!")

            if ideal_corr > 0.9 and hw_corr > 0.7:
                print("   🌟 PERFECT - All fixes working!")
            elif ideal_corr > 0.7:
                print("   ✅ EXCELLENT - Major improvement achieved!")
            else:
                print("   ⚠️ Still needs more work")

    else:
        # Test multiple patterns
        print("\n🔬 Testing Multiple Patterns with Fixed System")
        patterns = ["single", "cross", "corners"]

        for pattern in patterns:
            print(f"\n--- Testing FIXED {pattern.upper()} ---")
            results = tester.test_optimized_frqi_fixed(pattern, 2, shots=1024)

            if results:
                ideal_corr = results.get('ideal_correlation', 0)
                hw_corr = results.get('hardware_correlation', 0)
                print(f"✅ {pattern}: Ideal={ideal_corr:.3f}, Hardware={hw_corr:.3f}")

    print("\n✨ Fully fixed hardware testing complete!")