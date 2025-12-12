import logging
import random
import librosa
import numpy as np
from collections import deque
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pyftg import (AIInterface, AudioData, CommandCenter, FrameData, GameData,
                   Key, RoundResult, ScreenData)

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

ACTION_GROUPS = {
    "movement": https://www.ice.ci.ritsumei.ac.jp/~ftgaic/Downloadfiles/Brief%20table%20of%20ZEN's%20skills.pdf["FORWARD_WALK", "DASH", "BACK_STEP", "JUMP", "FOR_JUMP", "BACK_JUMP"],
    "guard": ["STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD"],
    "pokes": ["STAND_A", "STAND_B", "CROUCH_A", "CROUCH_B"],
    "anti_air": ["STAND_FB", "AIR_UA", "AIR_UB"],
    "fireball": ["STAND_D_DF_FA", "STAND_D_DF_FB", "AIR_D_DF_FA", "AIR_D_DF_FB"],
    "super": ["STAND_D_DF_FC"],
    "throw": ["THROW_A", "THROW_B"],
    "air_attack": ["AIR_A", "AIR_B", "AIR_DA", "AIR_DB", "AIR_FA", "AIR_FB"]
}

ALL_MOVES = [move for group in ACTION_GROUPS.values() for move in group]
ATTACK_MOVES = (
    ACTION_GROUPS["pokes"] +
    ACTION_GROUPS["anti_air"] +
    ACTION_GROUPS["fireball"] +
    ACTION_GROUPS["super"] +
    ACTION_GROUPS["throw"] +
    ACTION_GROUPS["air_attack"]
)

class MyAI(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = False
        self.prev_my_hp = None
        self.prev_opponent_hp = None
        self.last_reward = 0
        self.round_reward_total = 0
        self.last_state = None
        self.last_action = None
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.91
        self.epsilon = 1.0
        self.epsilon_min = 0.48
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-4
        self.combo_count = 0
        self.num_projectiles = 0
        self.state_dim = 168
        self.action_dim = len(ALL_MOVES)

        if os.path.exists("myai_model.keras"):
            self.model = keras.models.load_model("myai_model.keras")
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=keras.losses.MeanSquaredError()
            )
            logger.info("Loaded existing model from myai_model.keras")
        else:
            self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_dim)
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=keras.losses.MeanSquaredError()
        )
        return model

    def initialize(self, game_data: GameData, player_number: int):
        logger.info("8625AI initialize")
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number
        self.prev_my_hp = 400
        self.prev_opponent_hp = 400
        self.last_state = None
        self.last_action = None
        self.combo_count = 0
        self.last_reward = 0

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def input(self) -> Key:
        return self.key

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(self.frame_data, self.player)

        if not frame_data.empty_flag:
            my_hp = self.frame_data.get_character(self.player).hp
            opp_hp = self.frame_data.get_character(not self.player).hp

            if self.prev_my_hp is not None and self.prev_opponent_hp is not None:
                damage_dealt = self.prev_opponent_hp - opp_hp
                damage_received = self.prev_my_hp - my_hp

                if damage_dealt > 0:
                    self.last_reward += damage_dealt * 15

                if damage_received > 0:
                    self.last_reward -= damage_received * 5

                if damage_received == 0 and self.cc.get_skill_flag() and self.cc.get_skill_key() in ACTION_GROUPS["guard"]:
                    self.last_reward += 1

            self.prev_my_hp = my_hp
            self.prev_opponent_hp = opp_hp

            if self.combo_count > 1:
                self.last_reward += self.combo_count * 20
        
            if self.cc.get_skill_flag() and self.cc.get_skill_key() in ACTION_GROUPS["super"]:
                self.last_reward += 200
        else:
            self.last_reward = 0

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def get_screen_data(self, screen_data: ScreenData):
        self.screen_data = screen_data

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data

    def _audio_to_state(self, audio_data):
        mfcc = np.zeros(39, dtype=np.float32)
        spec = np.zeros(129, dtype=np.float32)
        if audio_data and audio_data.raw_data_bytes:
            try:
                audio_array = np.frombuffer(audio_data.raw_data_bytes, dtype=np.int16).astype(np.float32)
                mfcc = librosa.feature.mfcc(y=audio_array, sr=16000, n_mfcc=39)
                mfcc = np.mean(mfcc, axis=1)
                mmax = np.max(np.abs(mfcc))
                if mmax > 0:
                    mfcc /= mmax
            except Exception as e:
                logger.error(f"Failed to compute MFCC: {e}")

        return np.concatenate([mfcc, spec])

    def _choose_action(self, state_vec):
        if random.random() < self.epsilon:
            if random.random() < 0.9:
                attack_indices = [i for i, m in enumerate(ALL_MOVES) if m in ATTACK_MOVES]
                if attack_indices:
                    return random.choice(attack_indices)
            return random.randint(0, len(ALL_MOVES) - 1)
        q_values = self.model.predict(state_vec.reshape(1, -1), verbose=0)[0]
        max_q = np.max(q_values)
        candidates = np.where(q_values == max_q)[0]
        attack_candidates = [i for i in candidates if ALL_MOVES[i] in ATTACK_MOVES]
        if attack_candidates:
            return random.choice(attack_candidates)
        return random.choice(candidates)
    
    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return

        state_vec = self._audio_to_state(self.audio_data)
        action_idx = self._choose_action(state_vec)
        move = ALL_MOVES[action_idx]

        self.cc.command_call(move)
        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            self.key.empty()
            self.cc.skill_cancel()

        if self.last_state is not None and self.last_action is not None:
            self.memory.append((self.last_state, self.last_action, self.last_reward, state_vec))
            self._learn_from_memory()

        self.last_state = state_vec
        self.last_action = action_idx

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        targets = q_values.copy()
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.model.fit(states, targets, epochs=1, verbose=0)

    def round_end(self, round_result: RoundResult):

        if self.player == 1:
            my_hp = round_result.remaining_hps[0]
            opponent_hp = round_result.remaining_hps[1]
        else:
            my_hp = round_result.remaining_hps[1]
            opponent_hp = round_result.remaining_hps[0]
        self.combo_count = 0

        if my_hp == opponent_hp:
            outcome = "Draw"
            final_reward = -200
        elif my_hp > opponent_hp:
            outcome = "Win"
            final_reward = 800 + (my_hp - opponent_hp)
        else:
            outcome = "Loss"
            final_reward = -1500 - (opponent_hp - my_hp)

        self.last_reward = final_reward
        self.round_reward_total = 0
        try:
            self.model.save("myai_model.keras")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def game_end(self):
        print("Game ended")

    def close(self):
        pass