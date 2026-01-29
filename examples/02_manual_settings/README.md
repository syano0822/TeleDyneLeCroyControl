# 02_manual_settings

Configure oscilloscope channel, acquisition, and trigger settings from a JSON file, or save current scope settings to a file.

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
    "channels": {
      "1": { "state": "HIGH", "level": 0.1 },
      "2": { "state": "LOW", "level": -0.05 }
    },
    "mode": "SINGLE"
  }
}
```

### Channel Settings

| Field | Description |
|-------|-------------|
| `channels.<n>.vdiv` | Vertical scale (V/div) |
| `channels.<n>.offset` | Vertical offset (V) |
| `channels.<n>.enabled` | Enable/disable channel |

### Acquisition Settings

| Field | Description |
|-------|-------------|
| `acquisition.tdiv` | Horizontal scale (s/div) |
| `acquisition.sampling_period` | Sample interval (s) |

### Trigger Settings

| Field | Description |
|-------|-------------|
| `trigger.channels.<n>.state` | `HIGH`, `LOW`, or `DONT_CARE` |
| `trigger.channels.<n>.level` | Absolute trigger level (V) |
| `trigger.channels.<n>.level_offset` | Relative to baseline (V), used if `level` is not set |
| `trigger.mode` | `SINGLE`, `NORM`, or `AUTO` |

**Trigger States:**
- `HIGH` - Trigger on rising edge (high level)
- `LOW` - Trigger on falling edge (low level)
- `DONT_CARE` - Don't use this channel for triggering

## Running the Example

### Apply Settings from File

```bash
# Default: WavePro at 192.168.0.10, using settings.json
python -m examples.02_manual_settings.manual_settings

# Keep display on (default: off for faster acquisition)
python -m examples.02_manual_settings.manual_settings --display

# Specify model and address
python -m examples.02_manual_settings.manual_settings --model waverunner --address 192.168.1.100

# Use a custom config file
python -m examples.02_manual_settings.manual_settings --config my_settings.json
```

### Save Current Settings to File

```bash
# Save current scope settings to a JSON file
python -m examples.02_manual_settings.manual_settings --save current_settings.json

# With model/address
python -m examples.02_manual_settings.manual_settings --model wavepro --address 192.168.0.10 --save backup.json
```

## Expected Output

### When Applying Settings

```
Loading settings from: .../settings.json
Manual settings applied.
Display: ON
Channels: [1, 2]
TDIV: 0.001 s/div
Trigger:
  CH1: HIGH, level=0.1
  CH2: LOW, level=-0.05
Verify changes on the scope front panel.
```

### When Saving Settings

```
Reading current settings from scope...
Settings saved to: current_settings.json

Channels:
  CH1: 200.0 mV/div, offset=0.000 V [ON]
  CH2: 200.0 mV/div, offset=0.000 V [ON]
  CH3: 500.0 mV/div, offset=0.000 V [OFF]
  CH4: 500.0 mV/div, offset=0.000 V [OFF]

Acquisition:
  TDIV: 0.001 s/div

Trigger (SINGLE):
  CH1: HIGH, level=0.1000 V
  CH2: LOW, level=-0.0500 V
  CH3: DONT_CARE, level=0.0000 V
  CH4: DONT_CARE, level=0.0000 V
```

## Expected Behavior on the Scope

After running this script, verify on the oscilloscope front panel:

- **Channels 1 and 2**: Enabled with 200 mV/div vertical scale
- **Timebase**: 1 ms/div horizontal scale
- **Trigger**: Pattern trigger with CH1 HIGH (0.1V), CH2 LOW (-0.05V), single-shot mode

## Troubleshooting

- **JSON parse error**: Validate your settings.json syntax
- **Configuration errors**: Ensure the scope supports the requested settings
- **Display off**: Use `--display` flag to keep the display on
- **Trigger not working**: Check trigger levels are within signal range
