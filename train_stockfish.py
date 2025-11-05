import os
import time
import traceback
from deep_chess_core import DeepChessAgent, train_vs_stockfish

def safe_train_loop():
    """Vòng lặp huấn luyện liên tục với khả năng tự phục hồi."""
    print("=== 🚀 DeepChess Continuous Trainer started ===")

    # Tạo agent (có thể load lại model nếu có sẵn)
    agent = DeepChessAgent()
    if os.path.exists("deepchess_latest.pt"):
        try:
            agent.load("deepchess_latest.pt")
            print("📦 Model đã được tải lại từ checkpoint.")
        except Exception as e:
            print("⚠️ Không thể tải checkpoint:", e)

    total_games = 0
    while True:
        try:
            print(f"🎮 Huấn luyện batch mới... (tổng: {total_games} ván)")
            train_vs_stockfish(agent, adaptive=True, episodes=100)

            total_games += 100
            agent.save("deepchess_latest.pt")
            print(f"✅ Đã lưu model sau {total_games} ván!")

            print("💤 Nghỉ 1 giây trước vòng tiếp theo...")
            time.sleep(1)

        except KeyboardInterrupt:
            print("🛑 Dừng huấn luyện thủ công.")
            break

        except Exception as e:
            print("⚠️ Lỗi trong vòng huấn luyện:", e)
            traceback.print_exc()
            print("⏳ Chờ 10 giây và thử lại...")
            time.sleep(10)

if __name__ == "__main__":
    safe_train_loop()
