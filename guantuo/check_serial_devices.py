#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USB串口设备检测和故障排除工具
"""

import os
import subprocess
import glob
import getpass

def check_usb_devices():
    """检查USB设备"""
    print("🔍 检查USB设备...")
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        usb_devices = result.stdout.split('\n')
        
        serial_devices = []
        for line in usb_devices:
            if any(keyword in line.lower() for keyword in ['serial', 'ch340', 'ch341', 'ftdi', 'cp210', 'pl2303']):
                serial_devices.append(line.strip())
                print(f"  ✅ 找到串口设备: {line.strip()}")
        
        if not serial_devices:
            print("  ❌ 未找到串口转换器设备")
            print("     请检查:")
            print("     - USB设备是否正确连接")
            print("     - USB线是否为数据线（非充电线）")
            print("     - 设备是否需要安装驱动")
        
        return serial_devices
    except Exception as e:
        print(f"  ❌ 检查USB设备失败: {e}")
        return []

def check_serial_device_files():
    """检查串口设备文件"""
    print("\n📁 检查串口设备文件...")
    
    patterns = {
        'USB串口': '/dev/ttyUSB*',
        'ACM设备': '/dev/ttyACM*', 
        '系统串口': '/dev/ttyS[0-3]'  # 只检查前4个系统串口
    }
    
    found_devices = []
    
    for device_type, pattern in patterns.items():
        devices = glob.glob(pattern)
        if devices:
            print(f"  📡 {device_type}:")
            for device in sorted(devices):
                if os.path.exists(device):
                    try:
                        stat = os.stat(device)
                        permissions = oct(stat.st_mode)[-3:]
                        print(f"    ✅ {device} (权限: {permissions})")
                        found_devices.append(device)
                    except Exception as e:
                        print(f"    ❌ {device} (无法访问: {e})")
        else:
            print(f"  ❌ 未找到 {device_type}")
    
    return found_devices

def check_user_permissions():
    """检查用户权限"""
    print("\n🔐 检查用户权限...")
    username = getpass.getuser()
    
    try:
        import grp
        dialout_group = grp.getgrnam('dialout')
        user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
        
        if 'dialout' in user_groups:
            print(f"  ✅ 用户 {username} 在 dialout 组中")
            return True
        else:
            print(f"  ❌ 用户 {username} 不在 dialout 组中")
            print("     解决方法:")
            print(f"     1. 执行: sudo usermod -a -G dialout {username}")
            print("     2. 注销并重新登录")
            print("     3. 或者临时使用: sudo python3 your_script.py")
            return False
    except Exception as e:
        print(f"  ❌ 检查权限失败: {e}")
        return False

def check_drivers():
    """检查驱动模块"""
    print("\n🔧 检查驱动模块...")
    
    common_drivers = ['ch341', 'ftdi_sio', 'cp210x', 'pl2303', 'cdc_acm']
    loaded_drivers = []
    
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        loaded_modules = result.stdout
        
        for driver in common_drivers:
            if driver in loaded_modules:
                print(f"  ✅ {driver} 驱动已加载")
                loaded_drivers.append(driver)
            else:
                print(f"  ❌ {driver} 驱动未加载")
        
        if not loaded_drivers:
            print("  💡 尝试加载常见驱动:")
            print("     sudo modprobe ch341")
            print("     sudo modprobe ftdi_sio")
            print("     sudo modprobe cp210x")
    except Exception as e:
        print(f"  ❌ 检查驱动失败: {e}")
    
    return loaded_drivers

def check_dmesg():
    """检查系统日志"""
    print("\n📝 检查最近的系统日志...")
    
    try:
        # 尝试获取最近的USB和串口相关日志
        result = subprocess.run(['dmesg'], capture_output=True, text=True)
        if result.returncode != 0:
            print("  ❌ 无法读取系统日志 (可能需要sudo权限)")
            return
        
        dmesg_lines = result.stdout.split('\n')
        recent_lines = dmesg_lines[-50:]  # 最近50行
        
        relevant_lines = []
        keywords = ['usb', 'ttyUSB', 'ttyACM', 'ch340', 'ch341', 'ftdi', 'serial']
        
        for line in recent_lines:
            if any(keyword in line.lower() for keyword in keywords):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            print("  📄 相关的系统日志:")
            for line in relevant_lines[-10:]:  # 显示最近10条相关日志
                print(f"    {line}")
        else:
            print("  ❌ 未找到相关的系统日志")
            
    except Exception as e:
        print(f"  ❌ 检查系统日志失败: {e}")

def suggest_solutions(usb_devices, device_files, has_permissions):
    """提供解决建议"""
    print("\n💡 解决建议:")
    
    if not usb_devices:
        print("  🔌 USB设备问题:")
        print("     - 重新插拔USB设备")
        print("     - 尝试不同的USB端口")
        print("     - 检查USB线是否为数据线")
        print("     - 确认设备是否需要安装Windows驱动（在Linux下通常不需要）")
    
    if usb_devices and not device_files:
        print("  🔧 驱动问题:")
        print("     - 执行: sudo modprobe ch341")
        print("     - 执行: sudo modprobe ftdi_sio")
        print("     - 执行: sudo udevadm trigger")
        print("     - 重新插拔USB设备")
    
    if not has_permissions:
        print("  🔐 权限问题:")
        print("     - 执行: sudo usermod -a -G dialout $USER")
        print("     - 注销重新登录")
        print("     - 或临时使用sudo运行程序")
    
    if device_files:
        print("  ✅ 设备检测成功:")
        print(f"     - 可以尝试使用: {device_files[0]}")
        print("     - 在代码中修改port参数")

def main():
    print("=" * 60)
    print("    USB串口设备检测和故障排除工具")
    print("=" * 60)
    
    # 执行各项检查
    usb_devices = check_usb_devices()
    device_files = check_serial_device_files()
    has_permissions = check_user_permissions()
    check_drivers()
    check_dmesg()
    
    # 提供解决建议
    suggest_solutions(usb_devices, device_files, has_permissions)
    
    print("\n" + "=" * 60)
    if device_files and has_permissions:
        print("✅ 系统配置正常，可以尝试运行串口程序")
        print(f"建议使用设备: {device_files[0]}")
    else:
        print("⚠️  发现配置问题，请按照上述建议进行修复")
    print("=" * 60)


if __name__ == "__main__":
    main() 