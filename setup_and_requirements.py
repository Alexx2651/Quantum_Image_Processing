"""
Setup script for Quantum Image Processing Project
Run this first to install dependencies and test your environment
"""

import subprocess
import sys
import os


def install_requirements():
    """Install all required packages"""
    requirements = [
        "qiskit>=0.45.0",
        "qiskit-aer>=0.13.0",
        "qiskit-ibm-runtime>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "pillow>=9.0.0",
        "scipy>=1.10.0",
        "jupyter>=1.0.0"
    ]

    print("📦 Installing required packages...")
    for requirement in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"✅ Installed: {requirement}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {requirement}")
            return False

    return True


def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing package imports...")

    try:
        import qiskit
        print(f"✅ Qiskit version: {qiskit.__version__}")

        from qiskit_aer import AerSimulator
        print("✅ Qiskit Aer imported successfully")

        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")

        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")

        from PIL import Image
        print("✅ PIL imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_quantum_circuit():
    """Test basic quantum circuit functionality"""
    print("\n🔬 Testing quantum circuit creation...")

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        # Create simple test circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        # Test simulation
        simulator = AerSimulator()
        job = simulator.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts()

        print(f"✅ Quantum circuit test successful: {counts}")
        return True

    except Exception as e:
        print(f"❌ Quantum circuit test failed: {e}")
        return False


def setup_ibm_quantum():
    """Guide user through IBM Quantum setup"""
    print("\n🌐 IBM Quantum Account Setup")
    print("=" * 40)

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        # Check if account is already saved
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")
            print("✅ IBM Quantum account already configured!")

            # Test backend access
            backends = service.backends()
            print(f"✅ Access to {len(backends)} quantum backends confirmed")
            return True

        except Exception:
            print("⚠️ IBM Quantum account not configured")
            print("\nTo set up your IBM Quantum account:")
            print("1. Go to https://quantum-computing.ibm.com/")
            print("2. Create a free account or sign in")
            print("3. Copy your API token from the account settings")
            print("4. Run this command with your token:")
            print("   QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN_HERE')")
            print("\nExample setup code:")
            print("""
from qiskit_ibm_runtime import QiskitRuntimeService

# Replace 'your_token_here' with your actual IBM Quantum token
QiskitRuntimeService.save_account(
    channel="ibm_quantum", 
    token="your_token_here"
)
            """)
            return False

    except ImportError:
        print("❌ qiskit-ibm-runtime not installed properly")
        return False


def create_project_structure():
    """Create recommended project directory structure"""
    print("\n📁 Creating project structure...")

    directories = [
        "quantum_image_processing",
        "quantum_image_processing/src",
        "quantum_image_processing/src/algorithms",
        "quantum_image_processing/src/testing",
        "quantum_image_processing/data",
        "quantum_image_processing/data/input_images",
        "quantum_image_processing/data/results",
        "quantum_image_processing/notebooks",
        "quantum_image_processing/docs"
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {e}")

    # Create requirements.txt file
    requirements_content = """# Quantum Image Processing Requirements
qiskit>=0.45.0
qiskit-aer>=0.13.0
qiskit-ibm-runtime>=0.15.0
numpy>=1.24.0
matplotlib>=3.6.0
pillow>=9.0.0
scipy>=1.10.0
jupyter>=1.0.0

# Optional but recommended
opencv-python>=4.7.0
scikit-image>=0.20.0
seaborn>=0.12.0
"""

    try:
        with open("quantum_image_processing/requirements.txt", "w") as f:
            f.write(requirements_content)
        print("✅ Created requirements.txt")
    except Exception as e:
        print(f"❌ Failed to create requirements.txt: {e}")


def run_quick_demo():
    """Run a quick demonstration of the quantum image processing"""
    print("\n🚀 Running quick quantum image processing demo...")

    try:
        # Quick FRQI demo
        from qiskit import QuantumCircuit
        import numpy as np

        # Create 2x2 test image
        test_image = np.array([[0.0, 1.0], [1.0, 0.0]])
        print(f"Test image:\n{test_image}")

        # Simple FRQI encoding demo
        qc = QuantumCircuit(3)  # 2 position qubits + 1 color qubit

        # Create superposition of positions
        qc.h([0, 1])  # Position qubits

        # Apply controlled rotations for pixel values
        # This is a simplified version - see full implementation for complete FRQI
        qc.cry(np.pi / 2, 0, 2)  # Pixel (0,1) = 1.0
        qc.x(1)
        qc.cry(np.pi / 2, 1, 2)  # Pixel (1,0) = 1.0
        qc.x(1)

        print(f"✅ Created quantum circuit with {qc.num_qubits} qubits")
        print(f"   Circuit depth: {qc.depth()}")
        print("   Ready for quantum image processing!")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🎯 Quantum Image Processing Project Setup")
    print("=" * 50)

    print("This setup will:")
    print("1. Install required Python packages")
    print("2. Test quantum computing functionality")
    print("3. Guide you through IBM Quantum account setup")
    print("4. Create project directory structure")
    print("5. Run a quick demo")

    input("\nPress Enter to continue...")

    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Package installation failed. Please check your Python environment.")
        return False

    # Step 2: Test imports
    if not test_imports():
        print("❌ Package import test failed. Please check your installation.")
        return False

    # Step 3: Test quantum functionality
    if not test_quantum_circuit():
        print("❌ Quantum circuit test failed. Please check your Qiskit installation.")
        return False

    # Step 4: IBM Quantum setup
    ibm_setup = setup_ibm_quantum()
    if not ibm_setup:
        print("⚠️ Continue without IBM Quantum access (simulator only)")

    # Step 5: Create project structure
    create_project_structure()

    # Step 6: Quick demo
    if not run_quick_demo():
        print("⚠️ Demo failed, but setup can continue")

    # Final summary
    print("\n🎉 Setup Complete!")
    print("=" * 30)
    print("✅ All packages installed")
    print("✅ Quantum circuits working")
    print("✅ Project structure created")

    if ibm_setup:
        print("✅ IBM Quantum account configured")
    else:
        print("⚠️ IBM Quantum account needs manual setup")

    print("\nNext Steps:")
    print("1. If IBM Quantum isn't set up, follow the instructions above")
    print("2. Run the FRQI encoder: python frqi_encoder.py")
    print("3. Test edge detection: python quantum_edge_detection.py")
    print("4. Try hardware testing: python ibm_hardware_test.py")

    print("\n📚 Learning Resources:")
    print("- IBM Qiskit Textbook: https://qiskit.org/textbook/")
    print("- Quantum Image Processing Papers: arXiv:quant-ph")
    print("- IBM Quantum Experience: https://quantum-computing.ibm.com/")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to start quantum image processing!")
    else:
        print("\n❌ Setup incomplete. Please resolve issues and try again.")