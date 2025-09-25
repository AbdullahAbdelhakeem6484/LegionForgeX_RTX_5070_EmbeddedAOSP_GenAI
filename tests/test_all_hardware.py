#!/usr/bin/env python3

"""
Lenovo Legion 5 15IAX10 Hardware Test Suite
Comprehensive testing of all hardware components
"""

import subprocess
import sys
import os
import json
from datetime import datetime

class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

class HardwareTester:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
    
    def run_command(self, cmd, timeout=10):
        """Run a command and return result"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                  text=True, timeout=timeout)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def log(self, message, color=Colors.BLUE):
        print(f"{color}[INFO]{Colors.NC} {message}")
    
    def success(self, message):
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
        
    def warning(self, message):
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
        
    def error(self, message):
        print(f"{Colors.RED}❌ {message}{Colors.NC}")
    
    def header(self, title):
        print(f"\n{Colors.CYAN}{'='*50}{Colors.NC}")
        print(f"{Colors.CYAN}  {title}{Colors.NC}")
        print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
    
    def test_cpu(self):
        """Test CPU functionality"""
        self.header("CPU TEST: Intel Core Ultra 9-275HX")
        
        ret, out, err = self.run_command("grep 'model name' /proc/cpuinfo | head -1")
        if ret == 0 and "275HX" in out:
            self.success("Intel Core Ultra 9-275HX detected")
            self.results['cpu'] = {'status': 'pass', 'model': out.split(':')[1].strip()}
        else:
            self.warning("CPU model not as expected")
            self.results['cpu'] = {'status': 'warning', 'model': 'Unknown'}
        
        # Test core count
        ret, out, err = self.run_command("nproc")
        if ret == 0:
            cores = int(out.strip())
            if cores >= 20:  # 24 cores expected
                self.success(f"CPU cores: {cores}")
            else:
                self.warning(f"CPU cores less than expected: {cores}")
        
        return self.results['cpu']['status'] == 'pass'
    
    def test_memory(self):
        """Test memory configuration"""
        self.header("MEMORY TEST: 32GB RAM")
        
        ret, out, err = self.run_command("free -g | grep Mem")
        if ret == 0:
            total_gb = int(out.split()[1])
            if total_gb >= 30:  # ~32GB
                self.success(f"Memory: {total_gb}GB detected")
                self.results['memory'] = {'status': 'pass', 'total_gb': total_gb}
            else:
                self.warning(f"Memory less than expected: {total_gb}GB")
                self.results['memory'] = {'status': 'warning', 'total_gb': total_gb}
        else:
            self.error("Memory detection failed")
            self.results['memory'] = {'status': 'fail'}
        
        return self.results['memory']['status'] == 'pass'
    
    def test_nvidia_gpu(self):
        """Test NVIDIA RTX 5070"""
        self.header("NVIDIA RTX 5070 TEST")
        
        # Check hardware detection
        ret, out, err = self.run_command("lspci | grep -i 'nvidia.*2d18'")
        if ret == 0:
            self.success("NVIDIA RTX 5070 hardware detected")
            gpu_detected = True
        else:
            self.error("NVIDIA RTX 5070 not detected in PCI")
            gpu_detected = False
        
        # Check drivers
        driver_status = 'unknown'
        if self.run_command("command -v nvidia-smi")[0] == 0:
            self.success("NVIDIA drivers installed")
            
            ret, out, err = self.run_command("nvidia-smi")
            if ret == 0:
                self.success("NVIDIA GPU active")
                driver_status = 'active'
            else:
                self.warning("GPU in power-save mode (normal)")
                driver_status = 'power_save'
        else:
            self.error("NVIDIA drivers not found")
            driver_status = 'missing'
        
        # Check CUDA
        cuda_status = 'unknown'
        if self.run_command("command -v nvcc")[0] == 0:
            self.success("CUDA toolkit available")
            ret, out, err = self.run_command("nvcc --version")
            if ret == 0:
                cuda_version = "Unknown"
                for line in out.split('\n'):
                    if "release" in line:
                        cuda_version = line.split("release")[1].split(",")[0].strip()
                        break
                self.log(f"CUDA version: {cuda_version}")
                cuda_status = 'available'
        else:
            self.warning("CUDA toolkit not found")
            cuda_status = 'missing'
        
        self.results['nvidia'] = {
            'status': 'pass' if gpu_detected else 'fail',
            'hardware_detected': gpu_detected,
            'driver_status': driver_status,
            'cuda_status': cuda_status
        }
        
        return gpu_detected
    
    def test_intel_graphics(self):
        """Test Intel integrated graphics"""
        self.header("INTEL GRAPHICS TEST")
        
        ret, out, err = self.run_command("lspci | grep -i 'intel.*graphics'")
        if ret == 0:
            self.success("Intel graphics detected")
            
            # Check driver
            ret, out, err = self.run_command("lsmod | grep i915")
            if ret == 0:
                self.success("Intel i915 driver loaded")
                driver_loaded = True
            else:
                self.warning("Intel i915 driver not loaded")
                driver_loaded = False
            
            self.results['intel_graphics'] = {
                'status': 'pass',
                'driver_loaded': driver_loaded
            }
            return True
        else:
            self.error("Intel graphics not detected")
            self.results['intel_graphics'] = {'status': 'fail'}
            return False
    
    def test_display(self):
        """Test display configuration"""
        self.header("DISPLAY TEST: 15.1\" WQXGA OLED 165Hz")
        
        if not os.environ.get('DISPLAY'):
            self.warning("No display available (headless mode)")
            self.results['display'] = {'status': 'skip', 'reason': 'headless'}
            return True
        
        ret, out, err = self.run_command("xrandr")
        if ret == 0:
            self.success("Display system active")
            
            # Check resolution
            current_res = None
            high_refresh = False
            
            for line in out.split('\n'):
                if '*' in line:
                    current_res = line.split()[0]
                    break
            
            if current_res == "2560x1600":
                self.success(f"WQXGA resolution active: {current_res}")
            elif current_res:
                self.log(f"Current resolution: {current_res}")
            
            if "165.00" in out:
                self.success("165Hz refresh rate available")
                high_refresh = True
            
            self.results['display'] = {
                'status': 'pass',
                'resolution': current_res,
                'high_refresh': high_refresh
            }
            return True
        else:
            self.error("Display system not working")
            self.results['display'] = {'status': 'fail'}
            return False
    
    def test_network(self):
        """Test network interfaces"""
        self.header("NETWORK TEST")
        
        # Check Ethernet
        ret, out, err = self.run_command("ip link show | grep enp")
        ethernet_ok = ret == 0
        if ethernet_ok:
            self.success("Ethernet interface detected")
        else:
            self.warning("Ethernet interface not found")
        
        # Check WiFi
        ret, out, err = self.run_command("ip link show | grep wlp")
        wifi_ok = ret == 0
        if wifi_ok:
            self.success("WiFi interface detected")
            
            # Check connection
            ret, out, err = self.run_command("nmcli -t -f STATE g")
            if ret == 0 and "connected" in out:
                self.success("Network connected")
                connected = True
            else:
                self.warning("Network not connected")
                connected = False
        else:
            self.warning("WiFi interface not found")
            connected = False
        
        self.results['network'] = {
            'status': 'pass' if (ethernet_ok or wifi_ok) else 'fail',
            'ethernet': ethernet_ok,
            'wifi': wifi_ok,
            'connected': connected
        }
        
        return ethernet_ok or wifi_ok
    
    def test_audio(self):
        """Test audio system"""
        self.header("AUDIO TEST")
        
        ret, out, err = self.run_command("aplay -l")
        if ret == 0:
            self.success("Audio devices detected")
            card_count = out.count("card ")
            self.log(f"Audio cards: {card_count}")
            
            self.results['audio'] = {
                'status': 'pass',
                'card_count': card_count
            }
            return True
        else:
            self.error("No audio devices found")
            self.results['audio'] = {'status': 'fail'}
            return False
    
    def test_storage(self):
        """Test storage devices"""
        self.header("STORAGE TEST: Dual NVMe")
        
        ret, out, err = self.run_command("lsblk -d | grep nvme")
        if ret == 0:
            nvme_count = len(out.strip().split('\n'))
            self.success(f"NVMe drives detected: {nvme_count}")
            
            # Check total space
            ret, out, err = self.run_command("df -h / | tail -1")
            if ret == 0:
                size_info = out.split()[1]
                self.log(f"Root filesystem: {size_info}")
            
            self.results['storage'] = {
                'status': 'pass',
                'nvme_count': nvme_count
            }
            return True
        else:
            self.error("NVMe drives not detected")
            self.results['storage'] = {'status': 'fail'}
            return False
    
    def test_gaming_readiness(self):
        """Test gaming components"""
        self.header("GAMING READINESS TEST")
        
        components = {
            'steam': 'Steam gaming platform',
            'gamemode': 'GameMode performance',
            'vulkaninfo': 'Vulkan graphics API'
        }
        
        gaming_score = 0
        for cmd, desc in components.items():
            if self.run_command(f"command -v {cmd}")[0] == 0:
                self.success(f"{desc} available")
                gaming_score += 1
            else:
                self.warning(f"{desc} not found")
        
        # Check 32-bit support
        ret, out, err = self.run_command("dpkg --print-foreign-architectures | grep i386")
        if ret == 0:
            self.success("32-bit architecture support enabled")
            gaming_score += 1
        else:
            self.warning("32-bit support not enabled")
        
        self.results['gaming'] = {
            'status': 'pass' if gaming_score >= 2 else 'warning',
            'score': gaming_score,
            'total': 4
        }
        
        return gaming_score >= 2
    
    def generate_report(self):
        """Generate test report"""
        self.header("TEST REPORT")
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"Test Duration: {duration:.1f} seconds")
        print(f"Test Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'pass')
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        
        # Component status
        for component, result in self.results.items():
            status = result['status']
            if status == 'pass':
                self.success(f"{component.replace('_', ' ').title()}: Working")
            elif status == 'warning':
                self.warning(f"{component.replace('_', ' ').title()}: Warning")
            elif status == 'skip':
                self.log(f"{component.replace('_', ' ').title()}: Skipped")
            else:
                self.error(f"{component.replace('_', ' ').title()}: Failed")
        
        # Save detailed report
        report_file = f"hardware_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'test_time': end_time.isoformat(),
                'duration_seconds': duration,
                'results': self.results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': passed_tests / total_tests * 100
                }
            }, f, indent=2)
        
        self.log(f"Detailed report saved: {report_file}")
        
        # Final verdict
        if passed_tests == total_tests:
            print(f"\n{Colors.GREEN}🎉 ALL TESTS PASSED - SYSTEM PERFECT!{Colors.NC}")
        elif passed_tests >= total_tests * 0.8:
            print(f"\n{Colors.YELLOW}⚠️  MOSTLY WORKING - MINOR ISSUES{Colors.NC}")
        else:
            print(f"\n{Colors.RED}❌ MULTIPLE ISSUES FOUND{Colors.NC}")
        
        return passed_tests / total_tests

def main():
    print("🚀 LENOVO LEGION 5 15IAX10 HARDWARE TEST SUITE")
    print("=" * 60)
    
    tester = HardwareTester()
    
    # Run all tests
    tester.test_cpu()
    tester.test_memory()
    tester.test_nvidia_gpu()
    tester.test_intel_graphics()
    tester.test_display()
    tester.test_network()
    tester.test_audio()
    tester.test_storage()
    tester.test_gaming_readiness()
    
    # Generate report
    success_rate = tester.generate_report()
    
    return 0 if success_rate >= 0.8 else 1

if __name__ == "__main__":
    sys.exit(main())
