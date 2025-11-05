from flask import Flask, send_file, jsonify
import os, threading, time, datetime, requests

app = Flask(__name__)
MODEL_PATH = "/data/checkpoint.pth"
PING_URL = "https://deepchess-keepalive.onrender.com/"  # đổi thành URL thật sau khi deploy

# === API ROUTES ===
@app.route("/")
def index():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "status": "✅ DeepChess trainer online",
        "utc_time": now,
        "model_exists": os.path.exists(MODEL_PATH)
    })

@app.route("/download")
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    return jsonify({"error": "Model not found yet"}), 404

# === KEEP-ALIVE PING LOOP ===
def auto_ping():
    while True:
        try:
            requests.get(PING_URL, timeout=10)
            print("[KeepAlive] Ping sent successfully ✅")
        except Exception as e:
            print("[KeepAlive] Ping failed:", e)
        time.sleep(600)  # mỗi 10 phút ping 1 lần

# === MAIN ===
if __name__ == "__main__":
    threading.Thread(target=auto_ping, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
