"""
Setuppp script for the player re-identification system.
"""

import os
import subprocess
import sys
import platform


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n" + "=" * 50)
    print("Installing Dependencies")
    print("=" * 50)
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✓ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n" + "=" * 50)
    print("Creating Directories")
    print("=" * 50)
    
    directories = ["models", "data", "output"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    print("✓ Directories created successfully!")
    return True


def download_model():
    """Download the YOLO model."""
    print("\n" + "=" * 50)
    print("Downloading Model")
    print("=" * 50)
    
    model_path = "models/best.pt"
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return True
    
    try:
        print("Downloading player detection model...")
        subprocess.check_call([sys.executable, "download_model.py"])
        
        if os.path.exists(model_path):
            print("✓ Model downloaded successfully!")
            return True
        else:
            print("✗ Model download failed!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading model: {e}")
        print("Please download manually from:")
        print("https://drive.google.com/file/d/1-5fOSHOSB9UXYPenOoZNAMScrePVCMD/view")
        return False


def run_tests():
    """Run system tests."""
    print("\n" + "=" * 50)
    print("Running Tests")
    print("=" * 50)
    
    try:
        print("Running system tests...")
        subprocess.check_call([sys.executable, "test_system.py"])
        
        print("✓ Tests completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Tests failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 50)
    print("Usage Instructions")
    print("=" * 50)
    
    print("1. Place your video file in the data/ directory")
    print("   Example: data/15sec_input_720p.mp4")
    print()
    print("2. Run the system:")
    print("   python -m src.main data/your_video.mp4")
    print()
    print("3. Additional options:")
    print("   --output output/result.mp4    # Custom output path")
    print("   --save-comparison            # Save side-by-side comparison")
    print("   --show-detections           # Show raw detections")
    print()
    print("4. Examples:")
    print("   python examples.py          # Run usage examples")
    print("   python test_system.py       # Run system tests")
    print()
    print("5. VS Code Integration:")
    print("   - Use Ctrl+Shift+P and search for 'Tasks: Run Task'")
    print("   - Select from available tasks like 'Run Player Re-ID (Demo)'")
    print("   - Use F5 to run in debug mode")
    print()
    print("Output files will be saved to the output/ directory")


def main():
    """Main setup function."""
    print("Player Re-Identification System Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nSetup failed due to dependency installation issues.")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    
    # Download model
    model_success = download_model()
    
    # Run tests
    test_success = run_tests()
    
    # Print final status
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    print(f"✓ Python version: Compatible")
    print(f"✓ Dependencies: Installed")
    print(f"✓ Directories: Created")
    print(f"{'✓' if model_success else '✗'} Model: {'Downloaded' if model_success else 'Failed'}")
    print(f"{'✓' if test_success else '✗'} Tests: {'Passed' if test_success else 'Failed'}")
    
    if model_success and test_success:
        print("\n🎉 Setup completed successfully!")
        print_usage_instructions()
    else:
        print("\n⚠️  Setup completed with issues.")
        print("Please check the error messages above.")
        if not model_success:
            print("- Model download failed: Download manually or check internet connection")
        if not test_success:
            print("- Tests failed: Check dependencies and system compatibility")


if __name__ == "__main__":
    main()
