from __future__ import annotations

from typing import Literal


class VoiceInterface:
    """Accepts user input as typed text or spoken voice.

    Voice mode: records from microphone using webrtcvad for endpoint
    detection, then transcribes with OpenAI Whisper (local, CPU-capable).

    Text mode: reads a line from stdin — useful for CI, scripted demos,
    and testing without a microphone.
    """

    SAMPLE_RATE = 16000   # Hz — required by both webrtcvad and Whisper
    FRAME_DURATION = 30   # ms per VAD frame (10, 20, or 30 are valid)
    SILENCE_TIMEOUT = 1.0 # seconds of silence before stopping recording

    def __init__(self, config: dict) -> None:
        self._config = config
        self._default_mode: Literal["text", "voice"] = config.get("mode", "text")
        self._whisper_model = config.get("whisper_model", "base")
        self._vad_aggressiveness = config.get("vad_aggressiveness", 3)
        self._whisper = None  # lazy-loaded

    def listen(self, mode: Literal["text", "voice"] | None = None) -> str:
        """Return the next user command as a string.

        Args:
            mode: override the configured default mode for this call.
        """
        m = mode or self._default_mode
        if m == "voice":
            return self._listen_voice()
        return self._listen_text()

    # ------------------------------------------------------------------
    # Text input
    # ------------------------------------------------------------------

    @staticmethod
    def _listen_text() -> str:
        try:
            return input("Command> ").strip()
        except EOFError:
            return ""

    # ------------------------------------------------------------------
    # Voice input
    # ------------------------------------------------------------------

    def _listen_voice(self) -> str:
        import collections
        import struct

        import sounddevice as sd
        import webrtcvad

        vad = webrtcvad.Vad(self._vad_aggressiveness)
        frame_samples = int(self.SAMPLE_RATE * self.FRAME_DURATION / 1000)
        silence_frames_needed = int(self.SILENCE_TIMEOUT * 1000 / self.FRAME_DURATION)

        print("Listening... (speak now)")

        audio_frames: list[bytes] = []
        ring: collections.deque[bool] = collections.deque(maxlen=silence_frames_needed)
        speaking = False

        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=frame_samples,
            dtype="int16",
            channels=1,
        ) as stream:
            while True:
                frame_bytes, _ = stream.read(frame_samples)
                is_speech = vad.is_speech(bytes(frame_bytes), self.SAMPLE_RATE)
                ring.append(is_speech)

                if is_speech:
                    speaking = True

                if speaking:
                    audio_frames.append(bytes(frame_bytes))
                    if len(ring) == silence_frames_needed and not any(ring):
                        break

        if not audio_frames:
            return ""

        raw_audio = b"".join(audio_frames)
        # Convert raw PCM bytes to float32 numpy array for Whisper
        import numpy as np
        audio_np = (
            np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )

        return self._transcribe(audio_np)

    def _transcribe(self, audio: "np.ndarray") -> str:
        if self._whisper is None:
            import whisper
            print(f"Loading Whisper model '{self._whisper_model}'...")
            self._whisper = whisper.load_model(self._whisper_model)

        result = self._whisper.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()
        print(f"Heard: {text!r}")
        return text
