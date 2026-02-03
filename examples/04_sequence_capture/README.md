# 04_sequence_capture

Capture multiple waveform segments using the oscilloscope's sequence mode.

## What is Sequence Mode?

Sequence mode captures multiple triggered events in rapid succession, storing each as a separate segment. This is useful for:
- Capturing rare or intermittent events
- Recording burst signals with minimal dead time
- Analyzing timing variations across multiple triggers

## Prerequisites

- Python 3.10+
- pyVISA with appropriate backend
- numpy, matplotlib
- Network access to the oscilloscope
- Signal source connected to CH1 and/or CH2

## Running the Example

```bash
cd examples/04_sequence_capture

# Default: WavePro at 192.168.0.10, 100 segments
python sequence_capture.py

# Testing without a real signal (force trigger)
python sequence_capture.py --force --segments 10

# Specify model, address, segments, and output directory
python sequence_capture.py \
    --model waverunner \
    --address 192.168.1.100 \
    --segments 50 \
    --outdir ./sequences
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `wavepro` | Oscilloscope model (`wavepro` or `waverunner`) |
| `--address` | `192.168.0.10` | IP address of the oscilloscope |
| `--segments` | `100` | Number of segments to capture |
| `--outdir` | `.` | Output directory for plot files |
| `--force` | off | Use force trigger for testing without real signals |

## Expected Output

```
Arming for sequence capture...
Triggered.
CH1 segments: 100
CH1 stats: min=-0.150 V, max=0.152 V, mean=0.001 V, p2p=0.302 V
Saved plot: sequence_ch1_seg0.png
CH1 stats: min=-0.148 V, max=0.151 V, mean=0.002 V, p2p=0.299 V
Saved plot: sequence_ch1_seg1.png
CH1 stats: min=-0.149 V, max=0.150 V, mean=-0.001 V, p2p=0.299 V
Saved plot: sequence_ch1_seg2.png
CH2 segments: 100
...
```

## Expected Behavior on the Scope

After running:
1. The scope enters sequence mode (visible in acquisition settings)
2. The scope arms and displays segment counter (e.g., "0/100")
3. As triggers occur, the segment counter increments
4. When all segments are captured, acquisition stops
5. The scope displays the last captured segment

## Output Files

Only the first 3 segments per channel are plotted as a sanity check:
- `sequence_ch1_seg0.png`, `sequence_ch1_seg1.png`, `sequence_ch1_seg2.png`
- `sequence_ch2_seg0.png`, `sequence_ch2_seg1.png`, `sequence_ch2_seg2.png`

The full segment data is available in the `data` dictionary for further processing.

## Troubleshooting

- **Timeout waiting for trigger**: Use `--force` flag for testing without real signals, or ensure a signal is connected
- **Fewer segments than expected**: Check if the trigger rate is too slow
- **Memory errors**: Reduce `--segments` if the scope runs out of acquisition memory
