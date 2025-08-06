#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import struct
import time

def read_serial_data(port='/dev/ttyUSB0', baudrate=115200):
    """
    从串口读取数据并解析
    
    参数:
    port: 串口设备路径 (默认: /dev/ttyUSB0)
    baudrate: 波特率 (默认: 115200)
    """
    
    # 数据包格式
    FRAME_HEADER = bytes([0xAA, 0xAB, 0xAC])  # 帧头
    DATA_LENGTH = 512  # 数据长度（字节）
    DATA_POINTS = 256  # 数据点数量
    BYTES_PER_POINT = 2  # 每个数据点的字节数
    
    try:
        # 配置串口
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,  # 数据位：8
            parity=serial.PARITY_NONE,  # 奇偶校验：无
            stopbits=serial.STOPBITS_ONE,  # 停止位：1
            timeout=1  # 读取超时时间（秒）
        )
        
        print(f"串口已打开: {port}, 波特率: {baudrate}")
        print("等待数据...")
        
        buffer = bytearray()
        
        while True:
            # 读取串口数据
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)
                
                # 查找帧头
                while len(buffer) >= len(FRAME_HEADER):
                    header_index = buffer.find(FRAME_HEADER)
                    
                    if header_index == -1:
                        # 没有找到帧头，清除部分缓冲区
                        buffer = buffer[1:]
                        continue
                    
                    # 移除帧头之前的无效数据
                    if header_index > 0:
                        buffer = buffer[header_index:]
                    
                    # 检查是否有完整的数据包
                    total_packet_length = len(FRAME_HEADER) + DATA_LENGTH
                    if len(buffer) >= total_packet_length:
                        # 提取数据部分
                        data_start = len(FRAME_HEADER)
                        data_end = data_start + DATA_LENGTH
                        data_bytes = buffer[data_start:data_end]
                        
                        # 解析数据点
                        print("=" * 50)
                        print("收到数据包，解析数据点:")
                        print("=" * 50)
                        
                        data_points = []
                        for i in range(DATA_POINTS):
                            # 每个数据点2字节，使用大端序解析为无符号16位整数
                            byte_offset = i * BYTES_PER_POINT
                            point_bytes = data_bytes[byte_offset:byte_offset + BYTES_PER_POINT]
                            
                            if len(point_bytes) == 2:
                                # 解析为大端序无符号16位整数
                                value = struct.unpack('>H', point_bytes)[0]
                                data_points.append(value)
                                
                                # 每行打印10个数据点
                                if (i + 1) % 10 == 0:
                                    print(f"点{i-8:3d}-{i+1:3d}: {' '.join(f'{data_points[j]:5d}' for j in range(i-9, i+1))}")
                                elif i == DATA_POINTS - 1:
                                    # 打印最后不足10个的数据点
                                    remaining_start = (i // 10) * 10
                                    print(f"点{remaining_start+1:3d}-{i+1:3d}: {' '.join(f'{data_points[j]:5d}' for j in range(remaining_start, i+1))}")
                        
                        print("=" * 50)
                        print(f"数据解析完成，共{len(data_points)}个数据点")
                        print(f"最小值: {min(data_points)}, 最大值: {max(data_points)}")
                        print(f"平均值: {sum(data_points) / len(data_points):.2f}")
                        print("=" * 50)
                        print()
                        
                        # 移除已处理的数据包
                        buffer = buffer[total_packet_length:]
                    else:
                        # 数据包不完整，继续读取
                        break
            
            time.sleep(0.01)  # 短暂延时，避免过度占用CPU
    
    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")

def list_serial_ports():
    """列出可用的串口"""
    import serial.tools.list_ports
    import os
    import subprocess
    
    print("=== 串口设备检测 ===")
    
    # 检查USB设备
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        usb_devices = result.stdout
        print("\n检测到的USB设备:")
        for line in usb_devices.split('\n'):
            if 'serial' in line.lower() or 'ch340' in line.lower() or 'ch341' in line.lower() or 'ftdi' in line.lower():
                print(f"  🔍 {line}")
    except Exception as e:
        print(f"无法检查USB设备: {e}")
    
    # 检查可能的串口设备文件
    possible_ports = []
    port_patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ttyS*']
    
    print("\n可能的串口设备文件:")
    for pattern in port_patterns:
        try:
            import glob
            devices = glob.glob(pattern)
            if devices:
                for device in devices:
                    if os.path.exists(device):
                        possible_ports.append(device)
                        # 检查设备权限
                        stat = os.stat(device)
                        permissions = oct(stat.st_mode)[-3:]
                        print(f"  📁 {device} (权限: {permissions})")
        except Exception as e:
            print(f"检查 {pattern} 时出错: {e}")
    
    # 使用pyserial检测
    ports = serial.tools.list_ports.comports()
    print(f"\npyserial检测到的串口 (共{len(ports)}个):")
    usb_ports = []
    
    if ports:
        for port in ports:
            print(f"  📡 {port.device} - {port.description}")
            if port.hwid != 'n/a':
                print(f"      硬件ID: {port.hwid}")
            # 优先选择USB设备
            if 'USB' in port.device or 'ACM' in port.device:
                usb_ports.append(port.device)
    else:
        print("  ❌ 未找到可用串口")
    
    # 检查用户权限
    import getpass
    username = getpass.getuser()
    try:
        import grp
        dialout_group = grp.getgrnam('dialout')
        if username in dialout_group.gr_mem:
            print(f"\n✅ 用户 {username} 已在 dialout 组中")
        else:
            print(f"\n⚠️  用户 {username} 不在 dialout 组中")
            print("   解决方法: sudo usermod -a -G dialout $USER")
            print("   然后注销重新登录")
    except Exception as e:
        print(f"\n无法检查用户组: {e}")
    
    print("=" * 50)
    
    # 返回优先USB设备，否则返回所有设备
    return usb_ports if usb_ports else [port.device for port in ports]


if __name__ == "__main__":
    print("串口数据接收程序")
    print("数据格式: 帧头(AA AB AC) + 512字节数据(256个数据点)")
    print()
    
    # 列出可用串口
    available_ports = list_serial_ports()
    print()
    
    if available_ports:
        # 使用第一个可用串口，或者用户可以修改为具体的串口路径
        port = available_ports[0]
        print(f"将使用串口: {port}")
        print("如需使用其他串口，请修改代码中的port参数")
    else:
        # 默认串口路径（根据Linux系统调整）
        port = '/dev/ttyUSB0'
        print(f"使用默认串口: {port}")
        print("如果串口路径不正确，请按以下步骤检查:")
        print("1. 确保USB设备已连接并识别")
        print("2. 检查是否需要重新插拔USB设备")
        print("3. 确认用户在dialout组中: sudo usermod -a -G dialout $USER")
        print("4. 可能需要加载驱动: sudo modprobe ch341 或 sudo modprobe ftdi_sio")
    print()
    print("按Ctrl+C停止程序")
    print()
    
    # 开始读取数据
    read_serial_data(port=port, baudrate=115200)
