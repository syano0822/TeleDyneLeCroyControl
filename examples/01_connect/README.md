# 01_connect

Establish a basic connection to a Teledyne LeCroy oscilloscope and query its identity.

## Prerequisites

- Python 3.10+
- pyVISA with appropriate backend (NI-VISA or pyvisa-py)
- Network access to the oscilloscope

## Running the Example

```bash
# Default: WavePro at 192.168.0.10
python -m examples.01_connect.connect

# Specify model and address
python -m examples.01_connect.connect --model waverunner --address 192.168.1.100
```

## Expected Output

On success, you should see the instrument identification string:

```
Connected (wavepro) -> LECROY,WAVEPRO804HD,LCRY1234,8.2.0
```

The exact string depends on your oscilloscope model and firmware version.

## Expected Behavior on the Scope

The oscilloscope display remains unchanged. This example only queries the instrument identity without modifying any settings.

## Troubleshooting

- **Connection timeout**: Verify the IP address and that the scope is on the network
- **VISA errors**: Ensure pyVISA backend is properly installed
- **Permission errors**: Check firewall settings on both PC and oscilloscope
