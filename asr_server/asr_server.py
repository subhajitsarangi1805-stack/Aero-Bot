import asyncio, glob, json, os, sys, time
import websockets
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)
RATE = 16000
GRAMMAR = '["turn on the buzzer", "turn off the buzzer","buzzer on","buzzer off","[unk]"]'
GRAMMAR = None


def find_model():
    for p in ["model"] + sorted(glob.glob("vosk-model*")):
        if os.path.isdir(p):
            return p
    sys.exit("No Vosk model folder found next to asr_server.py")


def new_recognizer(model):
    return KaldiRecognizer(model, RATE, GRAMMAR) if GRAMMAR else KaldiRecognizer(model, RATE)


async def run(esp_ip):
    model = Model(find_model())
    uri = f"ws://{esp_ip}:81"
    while True:
        try:
            async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
                await ws.send(json.dumps({"role": "asr"}))
                print("Connected to", uri)
                rec, t_start, first = None, 0.0, True
                async for msg in ws:
                    if isinstance(msg, bytes):
                        if rec is None:
                            continue
                        rec.AcceptWaveform(msg)
                        if first:
                            first = False
                            ms = round((time.time() - t_start) * 1000)
                            await ws.send(json.dumps({"type": "asr_start", "ms": ms}))
                        continue
                    try:
                        d = json.loads(msg)
                    except ValueError:
                        continue
                    cmd = d.get("cmd")
                    if cmd == "start":
                        rec, first, t_start = new_recognizer(model), True, time.time()
                        await ws.send(json.dumps({"type": "ack"}))
                    elif cmd == "end" and rec is not None:
                        t = time.time()
                        text = json.loads(rec.FinalResult()).get("text", "")
                        ms = round((time.time() - t) * 1000)
                        print("ASR:", text or "(nothing heard)")
                        await ws.send(json.dumps({"type": "asr", "text": text, "ms": ms, "ok": True}))
                        rec = None
        except Exception as e:
            print("Disconnected, retrying in 2 s:", e)
            await asyncio.sleep(2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python asr_server.py <ESP32_IP>")
    asyncio.run(run(sys.argv[1]))