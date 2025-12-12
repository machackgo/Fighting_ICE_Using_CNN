try:
    from pyftg.models.key import Key
except Exception:
    from pyftg.models.structs.key import Key  # fallback for some pyftg layouts

from pynput import keyboard


class HumanKeyboardAgent:
    """Keyboard-controlled agent.

    Move:   W/A/S/D
    Attack: J/K/L  -> A/B/C
    Clear:  Q
    """

    def __init__(self):
        self.key = Key()
        self.pressed = set()
        self.frame_data = None
        self.non_delay_frame_data = None
        self.audio_data = None
        self.player_number = None

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def name(self):
        return "HUMAN"

    # --- Compatibility hooks for newer pyftg gateway (aio) ---
    def is_blind(self):
        return False

    def initialize(self, game_data, player_number):
        self.player_number = player_number

    def get_information(self, frame_data, is_control=True):
        self.frame_data = frame_data

    def get_non_delay_frame_data(self, frame_data):
        self.non_delay_frame_data = frame_data

    def get_audio_data(self, audio_data):
        self.audio_data = audio_data

    # Compatibility aliases (camelCase)
    def getInformation(self, frame_data, is_control=True):  # noqa: N802
        return self.get_information(frame_data, is_control)

    def getNonDelayFrameData(self, frame_data):  # noqa: N802
        return self.get_non_delay_frame_data(frame_data)

    def getAudioData(self, audio_data):  # noqa: N802
        return self.get_audio_data(audio_data)

    def processing(self):
        self.key = Key()

        # movement
        if "w" in self.pressed:
            setattr(self.key, "U", True)
        if "s" in self.pressed:
            setattr(self.key, "D", True)
        if "a" in self.pressed:
            setattr(self.key, "L", True)
        if "d" in self.pressed:
            setattr(self.key, "R", True)

        # attacks
        if "j" in self.pressed:
            setattr(self.key, "A", True)
        if "k" in self.pressed:
            setattr(self.key, "B", True)
        if "l" in self.pressed:
            setattr(self.key, "C", True)

    def input(self):
        return self.key

    def round_end(self, *args, **kwargs):
        self.pressed.clear()

    def game_end(self):
        self.pressed.clear()

    # Extra compatibility aliases some versions expect
    def roundEnd(self, *args, **kwargs):  # noqa: N802
        return self.round_end(*args, **kwargs)

    def gameEnd(self):  # noqa: N802
        return self.game_end()

    def _on_press(self, k):
        try:
            ch = k.char.lower()
        except Exception:
            return
        if ch == "q":
            self.pressed.clear()
        else:
            self.pressed.add(ch)

    def _on_release(self, k):
        try:
            ch = k.char.lower()
        except Exception:
            return
        self.pressed.discard(ch)

    def close(self):
        try:
            if self.listener is not None:
                self.listener.stop()
        except Exception:
            pass