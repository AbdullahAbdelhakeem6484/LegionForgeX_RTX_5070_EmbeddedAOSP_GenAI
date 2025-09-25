#!/usr/bin/env python3
"""
GPU Activation Test for Lenovo Legion 5 15IAX10
Tests various methods to activate the NVIDIA RTX 5070 GPU
"""

import subprocess
import sys
import time
import os

def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gpu_activation():
    """Test various GPU activation methods"""
    print("🚀 LENOVO LEGION 5 15IAX10 - GPU ACTIVATION TEST")
    print("=" * 60)
    
    # Test 1: Check current GPU state
    print("\n1️⃣ Current GPU State:")
    run_command("nvidia-smi", "NVIDIA System Management Interface")
    run_command("cat /sys/bus/pci/devices/0000:02:00.0/power_state", "GPU Power State")
    run_command("prime-select query", "PRIME Profile")
    
    # Test 2: Try to activate GPU with environment variables
    print("\n2️⃣ Testing GPU Activation Methods:")
    
    # Method 1: NVIDIA PRIME offload
    print("\n📋 Method 1: NVIDIA PRIME Offload")
    env = {
        '__NV_PRIME_RENDER_OFFLOAD': '1',
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'
    }
    
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            env={**os.environ, **env},
            capture_output=True, 
            text=True, 
            timeout=5
        )
        print(f"Result: {result.stdout.strip() if result.stdout else 'No output'}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 2: Force GPU wake-up with simple CUDA test
    print("\n📋 Method 2: CUDA Device Query")
    cuda_test = """
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("CUDA Devices Found: %d\\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\\n", i, prop.name);
    }
    
    return 0;
}
"""
    
    # Write and compile CUDA test
    with open('/tmp/cuda_test.cu', 'w') as f:
        f.write(cuda_test)
    
    compile_result = run_command(
        "nvcc /tmp/cuda_test.cu -o /tmp/cuda_test", 
        "Compiling CUDA test program"
    )
    
    if compile_result:
        run_command("/tmp/cuda_test", "Running CUDA device query")
    
    # Test 3: Check if GPU becomes available after CUDA call
    print("\n3️⃣ Post-Activation Check:")
    run_command("nvidia-smi", "NVIDIA System Management Interface (after CUDA)")
    
    # Test 4: Gaming/Graphics test
    print("\n4️⃣ Graphics API Test:")
    run_command("vulkaninfo --summary", "Vulkan API Information")
    
    # Cleanup
    try:
        os.remove('/tmp/cuda_test.cu')
        if os.path.exists('/tmp/cuda_test'):
            os.remove('/tmp/cuda_test')
    except:
        pass
    
    print("\n" + "=" * 60)
    print("🎯 GPU Activation Test Complete!")
    print("\n💡 Key Points:")
    print("• 'No devices were found' is NORMAL for hybrid graphics")
    print("• GPU activates automatically when needed")
    print("• Use __NV_PRIME_RENDER_OFFLOAD=1 for manual activation")
    print("• Gaming and CUDA workloads will wake the GPU")

if __name__ == "__main__":
    test_gpu_activation()