import serial
import time

# 串口通讯
serialPort = "COM4"  # 串口
baudRate = 9600  # 波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)
print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))


while(1):
    command = input()
    if command == 'z':
        break
    ser.write(command.encode())

time.sleep(3)
ser.write('f'.encode())
time.sleep(3)

results = 'R3 F2 L2 U3 D3 B2 L2 '
ser.write('R'.encode())
ser.write(results.encode())
while (1):
    str = ser.readline()
    print(str)
    time.sleep(0.5)