"""
IBM Quantum Hardware Testing for Quantum Image Processing
Updated for latest Qiskit Runtime API (2024/2025)
Tests FRQI encoding and QHED on real IBM quantum computers
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    print("✅ Using latest IBM Runtime API")
    RUNTIME_AVAILABLE = True
except ImportError:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        print("⚠️ Using legacy IBM Runtime API")
        RUNTIME_AVAILABLE = True
    except ImportError:
        print("❌ IBM Runtime not available")
        RUNTIME_AVAILABLE = False

from frqi_encoder import FRQIEncoder
from quantum_edge_detection import QuantumEdgeDetection

class IBMQuantumHardwareTester:
    """
    Test quantum image processing algorithms on real IBM Quantum hardware
    """

    def __init__(self):
        """Initialize IBM Quantum service connection"""
        if not RUNTIME_AVAILABLE:
            print("❌ IBM Runtime not available")
            self.service = None
            return

        try:
            # Initialize IBM Quantum service with modern API
            self.service = QiskitRuntimeService(channel="ibm_quantum")
            print("✅ Connected to IBM Quantum successfully!")

            # Get available backends
            self.backends = self.service.backends()
            operational_backends = [b for b in self.backends
                                  if b.status().operational and not b.configuration().simulator]
            print(f"Available quantum backends: {len(operational_backends)}")

        except Exception as e:
            print(f"❌ Failed to connect to IBM Quantum: {e}")
            print("Please set up your IBM Quantum account first:")
            print("QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
            self.service = None

    def select_best_backend(self, min_qubits=3):
        """
        Select the best available backend for our quantum image processing

        Args:
            min_qubits (int): Minimum qubits needed

        Returns:
            Backend: Best available quantum backend
        """
        if not self.service:
            return None

        # Filter backends by availability and qubit count
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

        # Sort by queue length (prefer less busy backends)
        available_backends.sort(key=lambda x: x['queue'])

        print("📋 Available Quantum Backends:")
        for i, backend_info in enumerate(available_backends[:5]):  # Show top 5
            print(f"{i+1}. {backend_info['name']}: {backend_info['qubits']} qubits, "
                  f"queue: {backend_info['queue']} jobs")

        selected = available_backends[0]['backend']
        print(f"🎯 Selected: {selected.name}")

        return selected

    def create_noise_model(self, backend):
        """
        Create a noise model based on real hardware characteristics

        Args:
            backend: IBM Quantum backend

        Returns:
            NoiseModel: Qiskit noise model for simulation
        """
        try:
            from qiskit_aer.noise import NoiseModel
            noise_model = NoiseModel.from_backend(backend)
            print(f"✅ Created noise model from {backend.name}")
            return noise_model
        except Exception as e:
            print(f"⚠️ Could not create noise model: {e}")
            return None

    def run_on_hardware(self, circuit, backend, shots=1024):
        """
        Execute quantum circuit on real IBM hardware with modern API

        Args:
            circuit (QuantumCircuit): Quantum circuit to execute
            backend: IBM Quantum backend
            shots (int): Number of shots

        Returns:
            dict: Execution results
        """
        print(f"🚀 Running on {backend.name} with {shots} shots...")

        try:
            # Ensure circuit has measurements
            if not circuit.cregs:
                circuit.measure_all()

            # Transpile circuit for the specific backend
            transpiled_circuit = transpile(circuit, backend=backend, optimization_level=2)
            print(f"📊 Circuit transpiled: {transpiled_circuit.depth()} depth, "
                  f"{transpiled_circuit.count_ops()} gates")

            # Create sampler with modern API
            try:
                # Try SamplerV2 with Session (latest approach)
                from qiskit_ibm_runtime import Session
                with Session(service=self.service, backend=backend) as session:
                    sampler = SamplerV2(session=session)
                    job = sampler.run([transpiled_circuit], shots=shots)

            except (NameError, ImportError, TypeError):
                try:
                    # Fallback: Try SamplerV2 without backend parameter
                    sampler = SamplerV2(backend=backend)
                    job = sampler.run([transpiled_circuit], shots=shots)
                except TypeError:
                    # Final fallback to legacy Sampler
                    print("⚠️ Using legacy Sampler API")
                    from qiskit_ibm_runtime import Sampler
                    sampler = Sampler(backend=backend)
                    job = sampler.run(transpiled_circuit, shots=shots)

            print(f"⏳ Job submitted: {job.job_id()}")
            print("Waiting for results...")

            # Wait for job completion
            start_time = time.time()
            try:
                result = job.result()
                execution_time = time.time() - start_time

                # Extract counts with proper error handling for different result formats
                try:
                    # Try modern result format
                    if hasattr(result[0].data, 'meas'):
                        raw_counts = result[0].data.meas.get_counts()
                    else:
                        raw_counts = result[0].data.get_counts()
                except (AttributeError, IndexError):
                    # Try legacy result format
                    raw_counts = result.get_counts()

                # Clean measurement strings (remove spaces)
                cleaned_counts = {}
                for state, count in raw_counts.items():
                    clean_state = state.replace(' ', '')
                    cleaned_counts[clean_state] = count

                print("✅ Job completed successfully!")

                return {
                    'counts': cleaned_counts,
                    'execution_time': execution_time,
                    'backend_name': backend.name,
                    'shots': shots,
                    'transpiled_depth': transpiled_circuit.depth(),
                    'transpiled_gates': transpiled_circuit.count_ops(),
                    'job_id': job.job_id()
                }

            except Exception as e:
                print(f"❌ Job execution failed: {e}")
                return None

        except Exception as e:
            print(f"❌ Hardware execution failed: {e}")
            return None

    def run_noisy_simulation(self, circuit, noise_model, shots=1024):
        """
        Run circuit on noisy simulator to compare with hardware

        Args:
            circuit (QuantumCircuit): Quantum circuit
            noise_model: Noise model from real hardware
            shots (int): Number of shots

        Returns:
            dict: Simulation results
        """
        print("🔧 Running noisy simulation...")

        try:
            # Add measurements if not present
            if not circuit.cregs:
                measured_circuit = circuit.copy()
                measured_circuit.measure_all()
            else:
                measured_circuit = circuit

            # Run noisy simulation
            simulator = AerSimulator(noise_model=noise_model)
            transpiled_circuit = transpile(measured_circuit, simulator)

            start_time = time.time()
            job = simulator.run(transpiled_circuit, shots=shots)
            result = job.result()
            execution_time = time.time() - start_time

            raw_counts = result.get_counts()

            # Clean measurement strings (remove spaces)
            cleaned_counts = {}
            for state, count in raw_counts.items():
                clean_state = state.replace(' ', '')
                cleaned_counts[clean_state] = count

            print("✅ Noisy simulation completed!")

            return {
                'counts': cleaned_counts,
                'execution_time': execution_time,
                'backend_name': 'noisy_simulator',
                'shots': shots,
                'transpiled_depth': transpiled_circuit.depth(),
                'transpiled_gates': transpiled_circuit.count_ops()
            }

        except Exception as e:
            print(f"❌ Noisy simulation failed: {e}")
            return None

    def run_ideal_simulation(self, circuit, shots=1024):
        """
        Run circuit on ideal simulator for comparison

        Args:
            circuit (QuantumCircuit): Quantum circuit
            shots (int): Number of shots

        Returns:
            dict: Ideal simulation results
        """
        print("⚡ Running ideal simulation...")

        try:
            # Add measurements if not present
            if not circuit.cregs:
                measured_circuit = circuit.copy()
                measured_circuit.measure_all()
            else:
                measured_circuit = circuit

            # Run ideal simulation
            simulator = AerSimulator()
            transpiled_circuit = transpile(measured_circuit, simulator)

            start_time = time.time()
            job = simulator.run(transpiled_circuit, shots=shots)
            result = job.result()
            execution_time = time.time() - start_time

            raw_counts = result.get_counts()

            # Clean measurement strings (remove spaces)
            cleaned_counts = {}
            for state, count in raw_counts.items():
                clean_state = state.replace(' ', '')
                cleaned_counts[clean_state] = count

            print("✅ Ideal simulation completed!")

            return {
                'counts': cleaned_counts,
                'execution_time': execution_time,
                'backend_name': 'ideal_simulator',
                'shots': shots,
                'transpiled_depth': transpiled_circuit.depth(),
                'transpiled_gates': transpiled_circuit.count_ops()
            }

        except Exception as e:
            print(f"❌ Ideal simulation failed: {e}")
            return None

    def analyze_hardware_vs_simulation(self, hardware_result, noisy_result, ideal_result):
        """
        Analyze differences between hardware and simulation results

        Args:
            hardware_result (dict): Results from real hardware
            noisy_result (dict): Results from noisy simulation
            ideal_result (dict): Results from ideal simulation

        Returns:
            dict: Analysis results
        """
        print("\n📊 Analyzing Hardware vs Simulation Results...")

        # Calculate fidelities and differences
        def calculate_fidelity(counts1, counts2, total_shots):
            """Calculate measurement fidelity between two count dictionaries"""
            all_states = set(counts1.keys()) | set(counts2.keys())
            fidelity = 0

            for state in all_states:
                p1 = counts1.get(state, 0) / total_shots
                p2 = counts2.get(state, 0) / total_shots
                fidelity += np.sqrt(p1 * p2)

            return fidelity

        analysis = {}

        if hardware_result and ideal_result:
            hw_ideal_fidelity = calculate_fidelity(
                hardware_result['counts'],
                ideal_result['counts'],
                hardware_result['shots']
            )
            analysis['hardware_ideal_fidelity'] = hw_ideal_fidelity
            print(f"🎯 Hardware vs Ideal Fidelity: {hw_ideal_fidelity:.3f}")

        if hardware_result and noisy_result:
            hw_noisy_fidelity = calculate_fidelity(
                hardware_result['counts'],
                noisy_result['counts'],
                hardware_result['shots']
            )
            analysis['hardware_noisy_fidelity'] = hw_noisy_fidelity
            print(f"🔧 Hardware vs Noisy Sim Fidelity: {hw_noisy_fidelity:.3f}")

        if noisy_result and ideal_result:
            noisy_ideal_fidelity = calculate_fidelity(
                noisy_result['counts'],
                ideal_result['counts'],
                ideal_result['shots']
            )
            analysis['noisy_ideal_fidelity'] = noisy_ideal_fidelity
            print(f"⚡ Noisy vs Ideal Fidelity: {noisy_ideal_fidelity:.3f}")

        return analysis

    def visualize_hardware_comparison(self, hardware_result, noisy_result, ideal_result, original_image):
        """
        Visualize comparison between hardware and simulation results

        Args:
            hardware_result (dict): Hardware execution results
            noisy_result (dict): Noisy simulation results
            ideal_result (dict): Ideal simulation results
            original_image (np.ndarray): Original test image
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Original image
        im1 = axes[0,0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('Original Image')
        plt.colorbar(im1, ax=axes[0,0])

        # Reconstruct images from quantum results
        image_size = original_image.shape[0]

        def reconstruct_from_counts(counts, shots):
            """Reconstruct image from measurement counts"""
            reconstructed = np.zeros((image_size, image_size))

            for state, count in counts.items():
                if len(state) >= 3:  # Ensure we have enough bits
                    color_bit = int(state[0])
                    position_bits = state[1:]

                    if len(position_bits) >= 2:  # For 2x2 image
                        position_idx = int(position_bits, 2)
                        row = position_idx // image_size
                        col = position_idx % image_size

                        if row < image_size and col < image_size and color_bit == 1:
                            reconstructed[row, col] += count / shots

            return reconstructed

        # Reconstruct images
        if ideal_result:
            ideal_reconstructed = reconstruct_from_counts(ideal_result['counts'], ideal_result['shots'])
            im2 = axes[0,1].imshow(ideal_reconstructed, cmap='hot', vmin=0, vmax=1)
            axes[0,1].set_title(f'Ideal Simulation\nTime: {ideal_result["execution_time"]:.3f}s')
            plt.colorbar(im2, ax=axes[0,1])

        if noisy_result:
            noisy_reconstructed = reconstruct_from_counts(noisy_result['counts'], noisy_result['shots'])
            im3 = axes[0,2].imshow(noisy_reconstructed, cmap='hot', vmin=0, vmax=1)
            axes[0,2].set_title(f'Noisy Simulation\nTime: {noisy_result["execution_time"]:.3f}s')
            plt.colorbar(im3, ax=axes[0,2])

        if hardware_result:
            hardware_reconstructed = reconstruct_from_counts(hardware_result['counts'], hardware_result['shots'])
            im4 = axes[0,3].imshow(hardware_reconstructed, cmap='hot', vmin=0, vmax=1)
            axes[0,3].set_title(f'Real Hardware\n{hardware_result["backend_name"]}\nTime: {hardware_result["execution_time"]:.3f}s')
            plt.colorbar(im4, ax=axes[0,3])

        # Measurement statistics comparison
        results_list = [ideal_result, noisy_result, hardware_result]
        titles = ['Ideal Simulator', 'Noisy Simulator', 'Real Hardware']

        for i, (result, title) in enumerate(zip(results_list, titles)):
            if result and i < 3:
                counts = result['counts']
                axes[1,i].bar(range(len(counts)), list(counts.values()))
                axes[1,i].set_title(f'{title}\nTotal Counts: {sum(counts.values())}')
                axes[1,i].set_xlabel('Quantum State')
                axes[1,i].set_ylabel('Count')

                # Rotate x-axis labels
                if len(counts) > 0:
                    axes[1,i].set_xticks(range(len(counts)))
                    axes[1,i].set_xticklabels(list(counts.keys()), rotation=45)

        # Execution time comparison
        if all(result for result in results_list):
            times = [result['execution_time'] for result in results_list]
            axes[1,3].bar(titles, times)
            axes[1,3].set_title('Execution Time Comparison')
            axes[1,3].set_ylabel('Time (seconds)')
            axes[1,3].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def run_complete_hardware_test(self, test_image, pattern_name="test"):
        """
        Run complete hardware test with all comparisons

        Args:
            test_image (np.ndarray): Test image
            pattern_name (str): Name of test pattern

        Returns:
            dict: Complete test results
        """
        print(f"\n🔬 Running complete hardware test for {pattern_name} pattern...")

        # Select backend
        backend = self.select_best_backend(min_qubits=3)
        if not backend:
            print("❌ No suitable backend available")
            return None

        # Create quantum edge detection circuit
        qed = QuantumEdgeDetection(image_size=test_image.shape[0])
        frqi_circuit = qed.frqi_encoder.encode_image(test_image)
        edge_circuit = qed.create_qhed_circuit(frqi_circuit)

        # Add measurements to circuit
        edge_circuit.measure_all()

        print(f"📋 Circuit stats: {edge_circuit.depth()} depth, {edge_circuit.count_ops()} gates")

        # Create noise model
        noise_model = self.create_noise_model(backend)

        # Run on all platforms
        shots = 1024

        # 1. Ideal simulation
        ideal_result = self.run_ideal_simulation(edge_circuit, shots)

        # 2. Noisy simulation
        noisy_result = None
        if noise_model:
            noisy_result = self.run_noisy_simulation(edge_circuit, noise_model, shots)

        # 3. Real hardware
        hardware_result = self.run_on_hardware(edge_circuit, backend, shots)

        # Analyze results
        analysis = self.analyze_hardware_vs_simulation(hardware_result, noisy_result, ideal_result)

        # Only visualize if we have valid results
        if ideal_result or hardware_result:
            try:
                self.visualize_hardware_comparison(hardware_result, noisy_result, ideal_result, test_image)
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
                print("Results are still valid, just visualization had issues")

        # Compile complete results
        complete_results = {
            'test_image': test_image,
            'pattern_name': pattern_name,
            'ideal_result': ideal_result,
            'noisy_result': noisy_result,
            'hardware_result': hardware_result,
            'analysis': analysis,
            'backend_name': backend.name if backend else None
        }

        return complete_results

# Main execution
if __name__ == "__main__":
    print("🚀 IBM Quantum Hardware Testing for Image Processing")
    print("=" * 60)

    # Initialize hardware tester
    tester = IBMQuantumHardwareTester()

    if not tester.service:
        print("❌ Cannot proceed without IBM Quantum access")
        print("Please set up your account and try again")
        exit()

    # Create test images
    encoder = FRQIEncoder(image_size=2)  # Start with 2x2 for hardware limitations

    test_patterns = ["edge", "corner"]  # Start with simple patterns

    all_hardware_results = {}

    for pattern in test_patterns:
        print(f"\n{'='*20} Testing {pattern.upper()} Pattern {'='*20}")

        # Create test image
        test_image = encoder.create_sample_image(pattern)
        print(f"Test image ({pattern}):\n{test_image}")

        # Run complete hardware test
        results = tester.run_complete_hardware_test(test_image, pattern)

        if results:
            all_hardware_results[pattern] = results

            # Print summary
            print(f"\n📋 {pattern.upper()} Pattern Results:")
            if results['ideal_result']:
                print(f"  Ideal execution time: {results['ideal_result']['execution_time']:.3f}s")
            if results['hardware_result']:
                print(f"  Hardware execution time: {results['hardware_result']['execution_time']:.3f}s")
                print(f"  Backend used: {results['hardware_result']['backend_name']}")

            if 'hardware_ideal_fidelity' in results['analysis']:
                print(f"  Hardware fidelity: {results['analysis']['hardware_ideal_fidelity']:.3f}")
        else:
            print(f"❌ {pattern} pattern test failed")

    # Final summary
    print(f"\n🎯 HARDWARE TESTING SUMMARY")
    print("=" * 60)

    if all_hardware_results:
        print("✅ Successfully tested quantum image processing on real IBM hardware!")
        print(f"Patterns tested: {list(all_hardware_results.keys())}")

        # Average fidelity
        fidelities = []
        for pattern, results in all_hardware_results.items():
            if 'hardware_ideal_fidelity' in results['analysis']:
                fidelities.append(results['analysis']['hardware_ideal_fidelity'])

        if fidelities:
            avg_fidelity = np.mean(fidelities)
            print(f"Average hardware fidelity: {avg_fidelity:.3f}")

        print("\n🔬 Next steps:")
        print("1. Test with larger images (4x4, 8x8) as hardware improves")
        print("2. Implement error mitigation techniques")
        print("3. Compare with classical edge detection performance")
        print("4. Explore other quantum image processing algorithms")

    else:
        print("❌ No successful hardware tests completed")
        print("This might be due to:")
        print("- Backend availability issues")
        print("- Account access limitations")
        print("- Network connectivity problems")
        print("Try again later or check your IBM Quantum account status")

    print(f"\n✨ Quantum image processing hardware test complete! ✨")