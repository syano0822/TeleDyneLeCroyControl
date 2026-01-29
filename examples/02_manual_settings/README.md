# 02_manual_settings

Configure oscilloscope channel, acquisition, and trigger settings from a JSON file.

## Prerequisites

- Python 3.10+
- pyVISA with appropriate backend
- Network access to the oscilloscope

## Configuration File

Edit `settings.json` to customize the oscilloscope configuration:

```json
{
  "channels": {
    "1": { "vdiv": 0.2, "offset": 0.0, "enabled": true },
    "2": { "vdiv": 0.2, "offset": 0.0, "enabled": true }
  },
  "acquisition": {
    "tdiv": 1e-3,
    "sampling_period": 1e-6
  },
  "trigger": {
    "source_channels": [1, 2],
    "slope": "RISING",
    "level_offset": 0.0,
    "mode": "SINGLE"
  }
}
```

| Field | Description |
|-------|-------------|
| `channels.<n>.vdiv` | Vertical scale (V/div) |
| `channels.<n>.offset` | Vertical offset (V) |
| `channels.<n>.enabled` | Enable/disable channel |
| `acquisition.tdiv` | Horizontal scale (s/div) |
| `acquisition.sampling_period` | Sample interval (s) |
| `trigger.source_channels` | Trigger source channel(s) |
| `trigger.slope` | `RISING` or `FALLING` |
| `trigger.level_offset` | Trigger level offset (V) |
| `trigger.mode` | `SINGLE`, `NORMAL`, or `AUTO` |

## Running the Example

```bash
# Default: WavePro at 192.168.0.10, using settings.json
python -m examples.02_manual_settings.manual_settings

# Specify model and address
python -m examples.02_manual_settings.manual_settings --model waverunner --address 192.168.1.100

# Use a custom config file
python -m examples.02_manual_settings.manual_settings --config my_settings.json
```

## Expected Output

```
Loading settings from: .../settings.json
Manual settings applied.
Channels: [1, 2]
TDIV: 0.001 s/div
Trigger: RISING edge on CH[1, 2]
Verify changes on the scope front panel.
```

## Expected Behavior on the Scope

After running this script, verify on the oscilloscope front panel:

- **Channels 1 and 2**: Enabled with 200 mV/div vertical scale
- **Timebase**: 1 ms/div horizontal scale
- **Trigger**: Rising edge, single-shot mode

## Troubleshooting

- **JSON parse error**: Validate your settings.json syntax
- **Configuration errors**: Ensure the scope supports the requested settings
- **No visible change**: Check that channels 1 and 2 have signals connected
- **SCPI errors**: Some settings may require specific scope modes to be enabled first
