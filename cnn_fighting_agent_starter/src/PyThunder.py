
from pyftg.models.attack_data import *
from pyftg.models import *
from pyftg.models.key import *
from pyftg.models.game_data import *
from pyftg.models.frame_data import *
from pyftg.models.audio_data import *
from pyftg.models.screen_data import *
from pyftg.models.round_result import *
from pyftg.aiinterface.command_center import *
from pyftg import AIInterface
import sys
import os
sys.path.append(os.path.dirname(__file__))


class PyThunder(AIInterface):

    def __init__(self):
        super().__init__()
        # self.commandsの順にコマンドをループ実行する
        self.commands = [
            "FOR_JUMP",
            "FOR_JUMP",
            "AIR_B",
            "STAND_D_DF_FC",
            "STAND_D_DF_FC",
        ]
        self.blind_flag = True

        # self.commandsの何個目の位置か記録する
        self.index: int = 0

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def initialize(self, game_data: GameData, player_number: bool):
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number
        self.game_data = game_data

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def input(self) -> Key:
        return self.key

    def get_information(self, frame_data: FrameData, is_control: bool, non_delay_frame_data: FrameData = None):
        # 前回条件分岐つきで調査用のコードを書いたけど、non_delay_frame_dataを使っていると勘違いされたので消した
        self.frame_data = frame_data
        self.cc.set_frame_data(self.frame_data, self.player)

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def get_audio_data(self, audio_data: AudioData):
        pass

    def processing(self):

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            self.index += 1
            if self.index == len(self.commands):
                self.index = 0
            self.key.empty()
            self.cc.skill_cancel()
            command = self.commands[self.index]
            self.cc.command_call(command)

    def round_end(self, round_result: RoundResult):
        print(self.__class__.name)
        for i in range(2):
            print(f"{i}\t{round_result.remaining_hps[i]}")
        if round_result.remaining_hps[0] > round_result.remaining_hps[1]:
            print("WIN")
        elif round_result.remaining_hps[0] < round_result.remaining_hps[1]:
            print("LOSE")
        else:
            print("DRAW")
        print("speed\t", round_result.elapsed_frame)
        print("*"*50)
        pass

    def game_end(self):
        pass

    def close(self):
        pass
