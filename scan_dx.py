from dynamixel_sdk import PortHandler, PacketHandler

port_name = '/dev/cu.usbserial-FT88Z15T'
protocols = [2.0, 1.0]
bauds = [1000000, 57142, 115200, 57600]

for prot in protocols:
    for baud in bauds:
        port = PortHandler(port_name)
        packet = PacketHandler(prot)
        ok_open = port.openPort()
        ok_baud = False
        if ok_open:
            ok_baud = port.setBaudRate(baud)
        print(f'prot={prot} baud={baud} open={ok_open} baud_ok={ok_baud}')
        if not ok_open or not ok_baud:
            try:
                port.closePort()
            except Exception:
                pass
            continue
        for dxl_id in range(11,16):
            try:
                model, result, error = packet.ping(port, dxl_id)
                print('  id', dxl_id, 'model', model, 'result', result, 'error', error)
            except Exception as e:
                print('  id', dxl_id, 'exception', e)
        port.closePort()
