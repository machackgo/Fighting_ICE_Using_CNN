import random

try:
    from pyftg.models.key import Key
except Exception:
    from pyftg.models.structs.key import Key  # fallback for some pyftg layouts


class KickAI:
    """Adaptive opponent with pyftg(aio) compatibility.

    Round 1: slower + fewer attacks.
    Later rounds: more aggressive.
    """

    def __init__(self):
        self.key = Key()
        self.frame_data = None
        self.non_delay_frame_data = None
        self.audio_data = None
        self.player_number = None

        self._t = 0
        self._cooldown = 0
        self.round_no = 1

    def name(self):
        return "KICK"

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

    def round_end(self, *args, **kwargs):
        self._cooldown = 0
        self.round_no += 1

    def game_end(self):
        self._cooldown = 0
        self.round_no = 1

    def roundEnd(self, *args, **kwargs):  # noqa: N802
        return self.round_end(*args, **kwargs)

    def gameEnd(self):  # noqa: N802
        return self.game_end()

    def _difficulty(self):
        return min(1.0, 0.20 + 0.35 * (self.round_no - 1))

    def _pick_frame(self):
        return self.non_delay_frame_data or self.frame_data

    def _get_char(self, fd, who):
        """Try multiple player-id conventions (1/2, 0/1, bool)."""
        if fd is None:
            return None

        getters = []
        for attr in ("get_character", "getCharacter"):
            if hasattr(fd, attr):
                getters.append(getattr(fd, attr))
        if not getters:
            return None

        candidates = [who]
        if isinstance(who, int):
            candidates += [who - 1, bool(who == 1)]

        for g in getters:
            for c in candidates:
                try:
                    ch = g(c)
                    if ch is not None:
                        return ch
                except Exception:
                    continue
        return None

    def _center_x(self, ch):
        if ch is None:
            return None
        for m in ("getCenterX", "getX"):
            if hasattr(ch, m):
                try:
                    return float(getattr(ch, m)())
                except Exception:
                    pass
        return None

    def _state(self, ch):
        if ch is None:
            return ""
        if hasattr(ch, "getState"):
            try:
                return str(ch.getState())
            except Exception:
                return ""
        return ""

    def _energy(self, ch):
        if ch is None:
            return 0
        if hasattr(ch, "getEnergy"):
            try:
                return int(ch.getEnergy())
            except Exception:
                return 0
        return 0

    def _clear_key(self):
        for b in ("U", "D", "L", "R", "A", "B", "C"):
            if hasattr(self.key, b):
                setattr(self.key, b, False)

    def processing(self):
        self._t += 1
        self.key = Key()
        self._clear_key()

        if self.player_number is None:
            return

        fd = self._pick_frame()
        if fd is None:
            # fallback so it never looks dead
            if random.random() < 0.55:
                setattr(self.key, random.choice(["L", "R"]), True)
            if random.random() < 0.20:
                setattr(self.key, random.choice(["A", "B"]), True)
            return

        me_id = self.player_number
        opp_id = 2 if me_id == 1 else 1

        me = self._get_char(fd, me_id)
        opp = self._get_char(fd, opp_id)

        mx = self._center_x(me)
        ox = self._center_x(opp)
        if mx is None or ox is None:
            # fallback if positions can't be parsed
            if random.random() < 0.55:
                setattr(self.key, random.choice(["L", "R"]), True)
            if random.random() < 0.20:
                setattr(self.key, random.choice(["A", "B"]), True)
            return

        dist = abs(ox - mx)
        opp_state = self._state(opp).lower()
        energy = self._energy(me)

        diff = self._difficulty()
        min_cd = int(22 - 12 * diff)
        max_cd = int(34 - 14 * diff)
        chase_dist = 210 - int(80 * diff)
        too_close = 60
        attack_gate = 0.20 + 0.65 * diff

        toward_right = (ox > mx)
        forward = "R" if toward_right else "L"
        back = "L" if toward_right else "R"

        if self._cooldown > 0:
            self._cooldown -= 1
            if random.random() < (0.10 + 0.35 * diff):
                setattr(self.key, forward, True)
            return

        if dist < too_close:
            setattr(self.key, back, True)
            self._cooldown = random.randint(min_cd, max_cd)
            return

        if dist > chase_dist:
            setattr(self.key, forward, True)
            self._cooldown = random.randint(min_cd, max_cd)
            return

        if random.random() > attack_gate:
            setattr(self.key, random.choice([forward, back]), True)
            self._cooldown = random.randint(min_cd, max_cd)
            return

        if "air" in opp_state or "jump" in opp_state:
            setattr(self.key, "U", True)
            setattr(self.key, "B", True)
            self._cooldown = random.randint(min_cd, max_cd)
            return

        r = random.random()
        if energy >= 150 and r < (0.05 + 0.25 * diff):
            setattr(self.key, "C", True)
        elif r < 0.60:
            setattr(self.key, "B", True)
        else:
            setattr(self.key, "A", True)

        if random.random() < (0.15 + 0.55 * diff):
            setattr(self.key, forward, True)

        self._cooldown = random.randint(min_cd, max_cd)

    def input(self):
        return self.key