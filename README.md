# LEDMatrix Audio Visualizer

A real-time USB microphone spectrum visualizer plugin for [ChuckBuilds/LEDMatrix](https://github.com/ChuckBuilds/LEDMatrix).

Designed for a 128x32 panel, but it sizes itself from LEDMatrix's active display dimensions.

## Features

- Real-time mono audio capture from the Raspberry Pi's default input or a selected USB microphone
- Logarithmic FFT spectrum display
- 8-64 configurable spectrum bars
- Sensitivity control
- Smoothing and falling peak markers
- Configurable low/high colours
- Optional **Hold while audio is active** mode
- Configurable activity threshold and release delay
- High-FPS LEDMatrix rendering
- Runtime config reload support
- Graceful failure if the microphone or PortAudio stack is unavailable

## Rotation takeover

`hold_while_audio_active` is **disabled by default**.

When enabled, the plugin reports live priority only while microphone RMS is above `activity_threshold`, plus the configured `release_delay_seconds`. This lets LEDMatrix temporarily keep the visualizer on-screen instead of continuing normal plugin rotation.

Sensitivity and activity detection are deliberately separate:

- `sensitivity` changes how strongly the FFT bars react visually.
- `activity_threshold` decides whether the incoming sound is active enough to hold the display.

This avoids a high visual sensitivity setting accidentally making room noise hold the display forever.

> LEDMatrix Vegas mode has its own live-content policy. If Vegas is configured to keep live content inside the ticker, its global policy still takes precedence.

## Requirements

The plugin uses:

- `numpy`
- `sounddevice`
- PortAudio on the Raspberry Pi

The plugin installer should install the Python packages in `requirements.txt`. On Raspberry Pi OS, make sure the PortAudio runtime is available:

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev
```

You can inspect detected audio devices with:

```bash
python3 -m sounddevice
```

For ALSA-level device checks:

```bash
arecord -l
```

## Install in LEDMatrix

In the LEDMatrix web UI:

1. Open **Plugin Manager**.
2. Choose **Install from GitHub** / **Install Single Plugin**.
3. Enter:

```text
https://github.com/jakejayeden/ledmatrix-audio-visualizer
```

The equivalent API call is:

```bash
curl -X POST http://your-pi-ip:5000/api/v3/plugins/install-from-url \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/jakejayeden/ledmatrix-audio-visualizer"}'
```

This first release targets LEDMatrix 3.8.0 or newer.

## Recommended starting settings for 128x32

```json
{
  "audio-visualizer": {
    "enabled": true,
    "input_device": null,
    "sensitivity": 1.0,
    "hold_while_audio_active": false,
    "activity_threshold": 0.025,
    "release_delay_seconds": 2.5,
    "sample_rate": 44100,
    "fft_size": 2048,
    "min_frequency": 60,
    "max_frequency": 16000,
    "bar_count": 32,
    "smoothing": 0.65,
    "peak_decay": 0.8,
    "show_peaks": true,
    "color_low": [0, 255, 80],
    "color_high": [255, 30, 0],
    "display_duration": 15
  }
}
```

## Selecting the USB microphone

Leave `input_device` blank/null first. `sounddevice` will use the Pi's default input device.

If the wrong microphone is selected, run:

```bash
python3 -m sounddevice
```

Then set `input_device` to either the input device number or its device-name substring. Example:

```json
"input_device": 2
```

or:

```json
"input_device": "USB Audio Device"
```

Changing the input device, sample rate or FFT size from the LEDMatrix UI restarts the capture stream automatically.

## Tuning sensitivity

Start with:

- `sensitivity`: `1.0`
- `activity_threshold`: `0.025`

If the bars barely move, raise `sensitivity` gradually. If quiet room noise causes the display-hold option to stay active, increase `activity_threshold` rather than reducing visual sensitivity.

Typical activity-threshold values will depend heavily on the USB mic's gain. The current RMS level is exposed through the plugin's info/status data to make calibration easier.

## Audio processing

The PortAudio callback does only lightweight work:

1. Copy samples into a rolling buffer.
2. Compute microphone RMS for activity detection.

FFT analysis is performed by the display render path, not inside the audio callback. The visualizer applies a Hann window, real FFT, logarithmic frequency bands, dB scaling and temporal smoothing before drawing the bars.

## Troubleshooting

### Red X on the matrix

The visualizer draws a red X when the audio stream could not be opened. Check LEDMatrix logs:

```bash
sudo journalctl -u ledmatrix -n 100 | grep -i audio
```

Then verify the mic:

```bash
arecord -l
python3 -m sounddevice
```

### PortAudio library not found

Install it:

```bash
sudo apt install -y libportaudio2 portaudio19-dev
```

Then restart LEDMatrix.

### Bars react but are too low/high

Adjust `sensitivity`.

### Hold mode activates from background noise

Increase `activity_threshold`.

### Hold mode drops out during quiet parts of music

Increase `release_delay_seconds` slightly, for example from `2.5` to `4.0`.

## Current status

Version `0.1.0` is the initial hardware-test release. It is intended to be tested on a Raspberry Pi with a real USB microphone and LEDMatrix 3.8+ before a 1.0 release.

## License

MIT
