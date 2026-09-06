from collections import deque
import random
class RemoteMemory():
    def __init__(self, *args):
        super().__init__(*args)
        self.memory = deque(maxlen=1200)
    # 5要素のtransition（state, action, reward, next_state, done）を追加
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    # ランダムにバッチ数サンプリング
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    #左から要素を取得する
    def get(self):
        try:
            item=self.memory.popleft()
        except IndexError:
            item=None
        return item
