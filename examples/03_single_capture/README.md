# 03_single_capture

Capture a single waveform from the oscilloscope and save it as a PNG plot.

## Prerequisites

- Python 3.10+
- pyVISA with appropriate backend
- numpy, matplotlib
- Network access to the oscilloscope
- Signal source connected to CH1 and/or CH2

## Running the Example

```bash
# Default: WavePro at 192.168.0.10, output to current directory
python -m examples.03_single_capture.single_capture

# Specify model, address, and output directory
python -m examples.03_single_capture.single_capture --model waverunner --address 192.168.1.100 --outdir ./captures
```

## Expected Output

```
Arming for single capture...
Triggered.
CH1 stats: min=-0.150 V, max=0.152 V, mean=0.001 V, p2p=0.302 V
Saved plot: single_capture_ch1.png
CH2 stats: min=-0.148 V, max=0.149 V, mean=-0.002 V, p2p=0.297 V
Saved plot: single_capture_ch2.png
```

## Expected Behavior on the Scope

After running:
1. The scope arms and displays "Armed" or "Ready" status
2. Once triggered, the scope captures and displays the waveform
3. The "STOP" indicator appears after acquisition completes
4. Waveform data is transferred to the PC and saved as PNG plots

## Output Files

- `single_capture_ch1.png` - Waveform plot for Channel 1
- `single_capture_ch2.png` - Waveform plot for Channel 2

## Troubleshooting

- **Timeout waiting for trigger**: Ensure a signal is connected and the trigger level is appropriate
- **No data returned**: Check that channels are enabled in the configuration
- **Plot looks wrong**: Verify the vertical scale (vdiv) matches your signal amplitude
