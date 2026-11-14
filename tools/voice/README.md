# Cached voice sidecar

This is an optional, offline-safe narration path for the G1 sorting policy. The
30 Hz policy sends tiny JSON events only after committed actions. A separate
process maps those events to pre-generated clips. Audio, ElevenLabs, subprocess
players, and the Unitree SDK never run in `act()`.

## Generate the cache

ElevenLabs generation requests 16 kHz mono signed 16-bit PCM and wraps it in a
WAV, which is directly usable by the G1 SDK2 backend:

```bash
export ELEVENLABS_API_KEY='...'
python tools/voice/generate_clips.py \
  --voice-id YOUR_ELEVENLABS_VOICE_ID \
  --model eleven_multilingual_v2
```

`eleven_flash_v2_5` is also supported. The key is read only from the environment
and must never be committed. `clips/manifest.json` records the text SHA-256,
voice, model, format, timestamp, and output filename. Re-running is idempotent:
unchanged, existing clips are skipped.

For offline placeholder audio (not prize-quality ElevenLabs output):

```bash
python tools/voice/generate_clips.py --say
python tools/voice/generate_clips.py --dry-run --voice-id YOUR_VOICE_ID
```

The `rap_intro` entry reserves a filename but has `generate: false`. Replace it
with an original 15-second ElevenLabs Music result if time permits; do not prompt
for a living artist, protected song, or distinctive lyric imitation.

## Run it

Enable `voice.enabled: true` in an approved overlay/config only after the audio
rules and volume are confirmed. It is **false by default**. Start one sidecar:

```bash
# Silent/log validation
python tools/voice/sidecar.py --backend null

# PC speaker; auto-selects aplay, ffplay, or afplay
python tools/voice/sidecar.py --backend local

# G1 speaker through Unitree SDK2 Python
python tools/voice/sidecar.py --backend g1 --interface eth0 --volume 60
```

The G1 backend guarded-imports
`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`, initializes DDS with
`ChannelFactoryInitialize(0, interface)`, then calls
`AudioClient.PlayStream("aaltonomous_voice", stream_id, pcm_bytes)` in one-second
chunks. Clips must be 16 kHz, mono, signed 16-bit PCM WAV. Test quietly on the
exact EDU robot and firmware; if the SDK/service is missing, use PC or null.

The receiver uses a bounded priority queue, suppresses duplicate datagrams,
rate-limits starts, and has one playback worker so clips cannot overlap.
`episode_done` is highest priority. Killing the sidecar, never starting it, or
filling its buffers only drops narration; policy actions remain unchanged.

### Optional Compose service

Keep the sidecar behind a profile so normal `docker compose up` does not start
it. Localhost UDP between separate containers requires host networking for both
the policy and this service (or run the sidecar directly beside the policy on
the host). This is an additive snippet, not a required deployment change:

```yaml
services:
  voice-sidecar:
    profiles: [voice]
    image: aaltonomous-policy:latest
    network_mode: host
    command: >-
      python tools/voice/sidecar.py --backend local
      --clips-dir /app/tools/voice/clips
    volumes:
      - ./tools/voice/clips:/app/tools/voice/clips:ro
    restart: unless-stopped
```

Start it explicitly with `docker compose --profile voice up voice-sidecar`.
Leave it stopped for the normal silent policy. If the policy container does not
already use host networking, run the sidecar on the same host/process network
instead of silently changing the finals network topology.

## Why this qualifies for Best Use of ElevenLabs

The clips are generated through ElevenLabs TTS and are triggered by real robot
state: understood instruction, confirmed lift, committed placement, safe
abstention, and episode result. Speech explains autonomous decisions to judges
rather than decorating a splash screen. The cache and manifest prove the model,
voice, exact text, and hashes while keeping offline finals deterministic. Show
the generator/API workflow and manifest in the submission, list ElevenLabs
under “built with,” and confirm cached generation eligibility with the prize
desk.

## 30-second demo-video script

- **0–4 s:** Robot and bins in frame. Play the optional original intro sting.
  Overlay: “ElevenLabs voice, cached for offline finals.”
- **4–9 s:** Give/show the language instruction. Audio: “Instructions
  understood. I will narrate committed moves.”
- **9–18 s:** Show one lift and placement without cuts. Audio after lift:
  “Yellow cube secured.” After release: “Yellow cube. Left bin.”
- **18–24 s:** Show an ambiguous object left untouched. Audio: “Confidence is
  low. I am leaving this object alone.” Overlay: “Safe abstention, no guessing.”
- **24–30 s:** Wide shot of sorted bins. Audio: “Sorting complete.” Briefly show
  `manifest.json`; overlay: “No cloud dependency in the control loop.”

