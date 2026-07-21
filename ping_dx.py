from dynamixel_sdk import PortHandler, PacketHandler

port_name = '/dev/cu.usbserial-FT88Z15T'
prot = 2.0
port = PortHandler(port_name)
packet = PacketHandler(prot)

if not port.openPort():
    print('open failed')
else:
    ok = port.setBaudRate(1000000)
    print('baud ok', ok)
    for dxl_id in range(11,16):
        try:
            model, result, error = packet.ping(port, dxl_id)
            print('id', dxl_id, 'model', model, 'result', result, 'error', error)
        except Exception as e:
            print('id', dxl_id, 'exception', e)
    port.closePort()
