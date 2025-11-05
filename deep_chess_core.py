import os
import time
import math
import random
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stockfish import Stockfish

# ==========================
# ⚙️ MODEL
# ==========================
class DeepChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(773, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, 4672)  # số nước hợp lệ tối đa
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


class DeepChessAgent:
    def __init__(self, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = []
        self.checkpoint_path = "/data/checkpoint.pth"

        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print("✅ Model loaded from checkpoint.")
        else:
            print("⚠️ No checkpoint found, starting fresh model.")

    def predict(self, state):
        with torch.no_grad():
            policy, value = self.model(torch.tensor(state, dtype=torch.float32).to(self.device))
        return policy.cpu().numpy(), value.item()

    def train_step(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, targets = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        targets = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1).to(self.device)

        _, values = self.model(states)
        loss = self.loss_fn(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"🧠 Train step done | Loss: {loss.item():.5f}")

    def save(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print("💾 Checkpoint saved.")


# ==========================
# ♟️ CHESS UTILITIES
# ==========================
def board_to_input(board: chess.Board):
    arr = np.zeros(773)
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece:
            arr[i] = piece.piece_type * (1 if piece.color else -1)
    arr[-5:] = [
        board.turn,
        board.fullmove_number,
        board.halfmove_clock,
        board.is_check(),
        board.is_checkmate(),
    ]
    return arr


# ==========================
# 🤖 TRAIN VS STOCKFISH
# ==========================
def train_vs_stockfish(agent, adaptive=True, episodes=100):
    stockfish_path = os.path.join("stockfish", "stockfish-linux-x64")
    engine = Stockfish(stockfish_path, parameters={"Threads": 1, "Skill Level": 1})
    total_games = 0
    total_loss = 0.0
    skill_level = 1
    time_limit = 0.05

    for game_idx in range(episodes):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn:  # AI move
                state = board_to_input(board)
                policy, _ = agent.predict(state)
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves)
                board.push(move)
            else:  # Stockfish move
                engine.set_fen_position(board.fen())
                move = engine.get_best_move_time(int(time_limit * 1000))
                if move:
                    board.push(chess.Move.from_uci(move))
                else:
                    break

        result = board.result()
        value = 1 if result == "1-0" else -1 if result == "0-1" else 0
        agent.replay_buffer.append((board_to_input(board), value))
        total_games += 1
        total_loss += abs(value)

        if total_games % 50 == 0:
            agent.train_step()
            agent.save()

            # Điều chỉnh độ khó Stockfish nếu adaptive
            if adaptive:
                avg_loss = total_loss / total_games
                if avg_loss < 0.3 and skill_level < 20:
                    skill_level += 1
                    time_limit = min(time_limit + 0.02, 0.3)
                    print(f"🔥 Stockfish lên cấp! skill={skill_level} | time={time_limit:.2f}s")
                    engine.update_engine_parameters({
                        "Skill Level": skill_level,
                        "Threads": 1
                    })
                total_loss = 0
                total_games = 0

        if game_idx % 10 == 0:
            print(f"[Stockfish] Game {game_idx}/{episodes} | Replay: {len(agent.replay_buffer)}")

    print("✅ Training session complete.")
