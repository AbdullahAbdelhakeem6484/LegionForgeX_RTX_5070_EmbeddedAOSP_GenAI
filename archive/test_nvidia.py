#!/usr/bin/env python3

import subprocess
import sys
import os

def test_nvidia_gpu():
    print("Testing NVIDIA GPU...")
    
    # Try to run nvidia-smi with different approaches
    print("\n1. Testing basic nvidia-smi:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    
    # Try with environment variables
    print("\n2. Testing with PRIME offload:")
    env = os.environ.copy()
    env['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    try:
        result = subprocess.run(['nvidia-smi'], env=env, capture_output=True, text=True, timeout=10)
        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Check kernel modules
    print("\n3. Checking NVIDIA kernel modules:")
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        nvidia_modules = [line for line in result.stdout.split('\n') if 'nvidia' in line]
        for module in nvidia_modules:
            print(f"  {module}")
    except Exception as e:
        print(f"Error checking modules: {e}")
    
    # Check device files
    print("\n4. Checking NVIDIA device files:")
    nvidia_devices = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-modeset', '/dev/nvidia-uvm']
    for device in nvidia_devices:
        if os.path.exists(device):
            print(f"  {device}: EXISTS")
        else:
            print(f"  {device}: NOT FOUND")
    
    # Check PCI device
    print("\n5. Checking PCI device status:")
    try:
        result = subprocess.run(['lspci', '-v', '-s', '02:00.0'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error checking PCI device: {e}")

if __name__ == "__main__":
    test_nvidia_gpu()
