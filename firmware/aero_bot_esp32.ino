#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <math.h>

#define EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN 0
#include <Audio_Classification_-_Keyword_Spotting_inferencing.h>

const char* WIFI_SSID     = "    ";
const char* WIFI_PASSWORD = "     ";

#define BUZZER_PIN 7

#define SAMPLE_RATE      16000
#define WINDOW_SAMPLES   EI_CLASSIFIER_RAW_SAMPLE_COUNT
#define SLICE_SAMPLES    4000
#define VAD_THRESHOLD    800.0f
#define VAD_HANG_SLICES  4
#define WAKE_THRESHOLD   0.80f
#define BUZZER_TIME      250
const char* WAKE_LABEL   = "helloworld";
const char* WAKE_DISPLAY = "HELLO WORLD";

#define PREROLL_MS       500
#define PREROLL_SAMPLES  (SAMPLE_RATE / 1000 * PREROLL_MS)
#define RING_SAMPLES     32768
#define RING_MASK        (RING_SAMPLES - 1)
#define CHUNK_SAMPLES    512

#define END_SILENCE_MS        900
#define NO_SPEECH_TIMEOUT_MS  2500
#define MAX_STREAM_MS         6000
#define ASR_TIMEOUT_MS        5000
#define COOLDOWN_MS           600

#define MODEL_SIZE_KB 0

#if WINDOW_SAMPLES > 30000
#error "Model window too large for ring buffer"
#endif

WebSocketsServer webSocket(81);

enum State { ST_LISTEN, ST_WAKE, ST_STREAM, ST_ASR, ST_COMMAND };
const char* STATE_NAME[] = {"LISTENING", "WAKE", "STREAMING", "ASR", "COMMAND"};
State state = ST_LISTEN;

int16_t ring[RING_SAMPLES];
volatile uint32_t ringTotal = 0;
uint32_t lastSliceTotal = 0, winStart = 0, streamPos = 0, wakeTotal = 0;

int asrClient = -1, micClient = -1;
bool buzzerLatched = false;
bool streaming = false, flushDone = false, heardSpeech = false, reportPending = false;
bool vadState = false, buzzerState = false, lightState = false;
int vadHang = 0;
float lastRMS = 0, lastWakeConfPct = 0, wakeInfMs = 0;

uint32_t wakeAt = 0, startSentAt = 0, flushStart = 0, lastVoiceAt = 0;
uint32_t cooldownUntil = 0, buzzerUntil = 0, asrDeadline = 0, commandUntil = 0, lastReconnect = 0;
int32_t wakeMs = -1, bufferMs = -1, netMs = -1, asrStartMs = -1, wakeToAsrMs = -1;

uint32_t detected = 0, falseCount = 0, missedCount = 0;
uint32_t pktCount = 0, droppedCount = 0, lastPktCount = 0;
uint32_t busyUs = 0;
uint32_t tStatus = 0, tRes = 0, lastRingSeen = 0;
bool micOK = false;
float cpuPct = 0, pps = 0;

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
    if (millis() - start > 30000) { Serial.println("\nWiFi connection failed"); return; }
  }
  Serial.println("\nWiFi connected");
  Serial.print("ESP32 IP: "); Serial.println(WiFi.localIP());
  Serial.print("Dashboard URL: ws://"); Serial.print(WiFi.localIP()); Serial.println(":81");
}

void buzzerOn()  { digitalWrite(BUZZER_PIN, HIGH); buzzerState = true; buzzerUntil = millis() + BUZZER_TIME; }
void buzzerOff() { digitalWrite(BUZZER_PIN, LOW);  buzzerState = false; }
void buzzerLatch(bool on) { buzzerLatched = on; if (on) { digitalWrite(BUZZER_PIN, HIGH); buzzerState = true; } else buzzerOff(); }

void emit(JsonDocument& doc, int target = -1) {
  static char out[560];
  size_t n = serializeJson(doc, out, sizeof(out));
  if (!n) return;
  if (target >= 0) { webSocket.sendTXT((uint8_t)target, out, n); return; }
  for (uint8_t i = 0; i < WEBSOCKETS_SERVER_CLIENT_MAX; i++)
    if ((int)i != asrClient && webSocket.clientIsConnected(i)) webSocket.sendTXT(i, out, n);
}

void setNum(JsonDocument& d, const char* k, int32_t v) { if (v < 0) d[k] = nullptr; else d[k] = v; }

void sendStatus(int target = -1) {
  StaticJsonDocument<320> d;
  d["type"] = "status";
  d["device"] = true;
  d["wifi"] = (WiFi.status() == WL_CONNECTED);
  d["ip"] = WiFi.localIP().toString();
  d["uptime_s"] = millis() / 1000;
  d["mic"] = micOK;
  d["kws"] = true;
  d["light"] = lightState;
  d["buzzer"] = buzzerState;
  d["src"] = "laptop";
  emit(d, target);
}

void sendState(int target = -1) {
  StaticJsonDocument<192> d;
  d["type"] = "state";
  d["state"] = STATE_NAME[state];
  d["keyword"] = WAKE_DISPLAY;
  if (state == ST_LISTEN) d["confidence"] = nullptr; else d["confidence"] = lastWakeConfPct;
  emit(d, target);
}

void setState(State s) { state = s; sendState(); }

void sendConf(float v) {
  StaticJsonDocument<96> d;
  d["type"] = "conf";
  d["value"] = v < 0 ? 0 : v;
  emit(d);
}

void sendResources(int target = -1) {
  StaticJsonDocument<384> d;
  uint32_t heapTotal = ESP.getHeapSize();
  uint32_t sketch = ESP.getSketchSize();
  d["type"] = "resources";
  d["ram_used_kb"] = (heapTotal - ESP.getFreeHeap()) / 1024;
  d["ram_total_kb"] = heapTotal / 1024;
  d["flash_used_kb"] = sketch / 1024;
  d["flash_total_kb"] = (sketch + ESP.getFreeSketchSpace()) / 1024;
  d["cpu"] = cpuPct;
  if (MODEL_SIZE_KB > 0) d["model_kb"] = MODEL_SIZE_KB; else d["model_kb"] = nullptr;
#ifdef EI_CLASSIFIER_TFLITE_LARGEST_ARENA_SIZE
  d["arena_kb"] = EI_CLASSIFIER_TFLITE_LARGEST_ARENA_SIZE / 1024;
#else
  d["arena_kb"] = nullptr;
#endif
  emit(d, target);
}

void sendNet(int target = -1) {
  StaticJsonDocument<192> d;
  d["type"] = "net";
  d["rssi"] = WiFi.RSSI();
  d["pps"] = pps;
  d["dropped"] = droppedCount;
  d["server"] = (asrClient >= 0);
  emit(d, target);
}

void sendStats(int target = -1) {
  StaticJsonDocument<192> d;
  uint32_t up = millis() / 1000;
  d["type"] = "stats";
  d["detected"] = detected;
  d["false"] = falseCount;
  d["missed"] = missedCount;
  d["false_per_hr"] = up > 60 ? falseCount * 3600.0f / up : 0.0f;
  emit(d, target);
}

void sendWake() {
  StaticJsonDocument<320> d;
  d["type"] = "wake";
  d["confidence"] = lastWakeConfPct;
  d["inference_ms"] = wakeInfMs;
  setNum(d, "wake_ms", wakeMs);
  setNum(d, "buffer_ms", bufferMs);
  setNum(d, "network_ms", netMs);
  setNum(d, "asr_start_ms", asrStartMs);
  setNum(d, "wake_to_asr_ms", wakeToAsrMs);
  d["preroll_ms"] = PREROLL_MS;
  emit(d);
  reportPending = false;
}

void sendSnapshot(int target) {
  sendStatus(target); sendState(target); sendResources(target); sendNet(target); sendStats(target);
}

int microphone_audio_signal_get_data(size_t offset, size_t length, float* out_ptr) {
  for (size_t i = 0; i < length; i++) out_ptr[i] = (float)ring[(winStart + offset + i) & RING_MASK];
  return 0;
}

float runKWS(uint32_t total) {
  winStart = total - WINDOW_SAMPLES;
  signal_t signal;
  signal.total_length = WINDOW_SAMPLES;
  signal.get_data = microphone_audio_signal_get_data;
  ei_impulse_result_t result = {0};

  uint32_t t0 = micros();
  EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);
  busyUs += micros() - t0;
  if (r != EI_IMPULSE_OK) return -1;

  wakeInfMs = result.timing.dsp + result.timing.classification + result.timing.anomaly;
  float v = 0;
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
    if (strcmp(result.classification[i].label, WAKE_LABEL) == 0) v = result.classification[i].value;
  return v;
}

void finishToListening() {
  streaming = false;
  vadHang = 0;
  cooldownUntil = millis() + COOLDOWN_MS;
  lastSliceTotal = ringTotal;
  setState(ST_LISTEN);
}

void triggerWake(uint32_t total, float conf, uint32_t sliceAt) {
  uint32_t now = millis();
  detected++;
  Serial.printf("[WAKE] conf=%.1f%% asrClient=%d\n", conf * 100.0f, asrClient);
  lastWakeConfPct = conf * 100.0f;
  wakeAt = now; wakeTotal = total; lastVoiceAt = now;
  wakeMs = now - sliceAt;
  bufferMs = netMs = asrStartMs = wakeToAsrMs = -1;
  flushDone = false; heardSpeech = false; reportPending = true;
  buzzerOn();
  setState(ST_WAKE);
  sendStats();
  if (asrClient >= 0) {
    startSentAt = millis();
    webSocket.sendTXT((uint8_t)asrClient, "{\"cmd\":\"start\",\"rate\":16000}");
    streamPos = total - PREROLL_SAMPLES;
    flushStart = millis();
    streaming = true;
  } else {
    streaming = false;
  }
}

void pumpStream() {
  if (!streaming || asrClient < 0) return;
  uint32_t total = ringTotal;
  if (total - streamPos > RING_SAMPLES - SLICE_SAMPLES) streamPos = total - (RING_SAMPLES - SLICE_SAMPLES);
  int16_t tmp[CHUNK_SAMPLES];
  for (uint8_t guard = 0; guard < 24 && (total - streamPos) >= CHUNK_SAMPLES; guard++) {
    for (int i = 0; i < CHUNK_SAMPLES; i++) tmp[i] = ring[(streamPos + i) & RING_MASK];
    if (webSocket.sendBIN((uint8_t)asrClient, (uint8_t*)tmp, sizeof(tmp))) pktCount++; else droppedCount++;
    streamPos += CHUNK_SAMPLES;
    if (!flushDone && (int32_t)(streamPos - wakeTotal) >= 0) { flushDone = true; bufferMs = millis() - flushStart; }
  }
}

void endStream() {
  pumpStream();
  streaming = false;
  if (asrClient >= 0) {
    webSocket.sendTXT((uint8_t)asrClient, "{\"cmd\":\"end\"}");
    asrDeadline = millis() + ASR_TIMEOUT_MS;
    setState(ST_ASR);
  } else {
    finishToListening();
  }
}

void onSlice(uint32_t total, uint32_t now) {
  double s = 0;
  for (int i = 0; i < SLICE_SAMPLES; i++) { int16_t v = ring[(total - SLICE_SAMPLES + i) & RING_MASK]; s += (double)v * v; }
  lastRMS = sqrt(s / SLICE_SAMPLES);
  bool voice = lastRMS >= VAD_THRESHOLD;
  vadState = voice;

  if (voice) {
    lastVoiceAt = now;
    if ((state == ST_WAKE || state == ST_STREAM) && now - wakeAt > 200) heardSpeech = true;
  }
  if (state != ST_LISTEN) return;

  if (voice) vadHang = VAD_HANG_SLICES; else if (vadHang > 0) vadHang--;
  if (vadHang == 0 || now < cooldownUntil) { sendConf(0); return; }

  float c = runKWS(total);
  sendConf(c);
  if (c >= WAKE_THRESHOLD) triggerWake(total, c, now);
}

void handleAsrText(const char* text, int32_t ms, bool ok) {
  Serial.printf("[ASR] result ok=%d text=\"%s\"\n", ok, text);
  StaticJsonDocument<512> d;
  d["type"] = "asr";
  d["text"] = text;
  setNum(d, "ms", ms);
  d["ok"] = ok;
  emit(d);

  String t = String(text); t.toLowerCase();
  if (t.indexOf("light") >= 0) {
    if (t.indexOf(" off") >= 0) lightState = false;
    else if (t.indexOf(" on") >= 0) lightState = true;
    sendStatus();
  }
  if (t.indexOf("buzzer") >= 0) {
    if (t.indexOf(" off") >= 0 || t.indexOf("stop") >= 0) buzzerLatch(false);
    else if (t.indexOf(" on") >= 0) buzzerLatch(true);
    sendStatus();
  }
  if (state == ST_ASR) { commandUntil = millis() + 1500; setState(ST_COMMAND); }
}

void onWsEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
  if (type == WStype_DISCONNECTED) {
    if ((int)num == asrClient) { asrClient = -1; streaming = false; }
    if ((int)num == micClient) micClient = -1;
    return;
  }
  if (type == WStype_CONNECTED) { sendSnapshot(num); return; }
  if (type == WStype_BIN) {
    if ((int)num == micClient && length >= 2) {
      size_t n = length / 2;
      uint32_t t = ringTotal;
      for (size_t i = 0; i < n; i++) { int16_t v; memcpy(&v, payload + 2 * i, 2); ring[(t + i) & RING_MASK] = v; }
      ringTotal = t + n;
    }
    return;
  }
  if (type != WStype_TEXT) return;

  StaticJsonDocument<512> d;
  if (deserializeJson(d, payload, length)) return;

  if (d["role"] == "asr") { asrClient = num; sendNet(); return; }
  if (d["role"] == "mic") { micClient = num; return; }

  const char* msgType = d["type"] | "";
  if (!strcmp(msgType, "ack"))            { netMs = millis() - startSentAt; Serial.printf("[ASR] ack, network RTT %d ms\n", (int)netMs); }
  else if (!strcmp(msgType, "asr_start")) { asrStartMs = d["ms"] | 0; wakeToAsrMs = millis() - wakeAt; Serial.printf("[ASR] started, wake->ASR %d ms\n", (int)wakeToAsrMs); }
  else if (!strcmp(msgType, "asr"))       { handleAsrText(d["text"] | "", d["ms"] | -1, d["ok"] | true); }

  const char* cmd = d["cmd"] | "";
  if (!strcmp(cmd, "mark_false"))  { falseCount++;  sendStats(); }
  if (!strcmp(cmd, "mark_missed")) { missedCount++; sendStats(); }

  if (d["light"].is<const char*>()) { lightState = (String(d["light"].as<const char*>()) == "on"); sendStatus(); }
  if (d["buzzer"].is<const char*>()) {
    String b = d["buzzer"].as<const char*>();
    if (b == "test") buzzerOn(); else if (b == "on") buzzerLatch(true); else if (b == "off") buzzerLatch(false);
    sendStatus();
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  buzzerOff();

  setupWiFi();
  webSocket.begin();
  webSocket.onEvent(onWsEvent);
  Serial.println("Audio source: laptop mic over WebSocket");

  Serial.printf("KWS window: %d samples, labels: %d, pre-roll: %d ms\n",
                (int)WINDOW_SAMPLES, (int)EI_CLASSIFIER_LABEL_COUNT, PREROLL_MS);
}

void loop() {
  webSocket.loop();
  uint32_t now = millis();

  if (buzzerState && !buzzerLatched && now >= buzzerUntil) buzzerOff();
  if (WiFi.status() != WL_CONNECTED && now - lastReconnect > 10000) { lastReconnect = now; WiFi.reconnect(); }

  uint32_t total = ringTotal;
  if (total - lastSliceTotal >= SLICE_SAMPLES) { lastSliceTotal = total; onSlice(total, now); }

  if (state == ST_WAKE || state == ST_STREAM) {
    pumpStream();
    if (state == ST_WAKE && (flushDone || !streaming) && now - wakeAt >= 300) setState(ST_STREAM);
    if (state == ST_STREAM) {
      uint32_t elapsed = now - wakeAt;
      bool endSilence = heardSpeech && (now - lastVoiceAt > END_SILENCE_MS);
      bool noSpeech = !heardSpeech && elapsed > NO_SPEECH_TIMEOUT_MS;
      if (endSilence || noSpeech || elapsed > MAX_STREAM_MS) endStream();
    }
  }
  if (state == ST_ASR && now > asrDeadline) { handleAsrText("", -1, false); finishToListening(); }
  if (state == ST_COMMAND && now >= commandUntil) finishToListening();

  if (reportPending && (wakeToAsrMs >= 0 || now - wakeAt > 2500)) sendWake();

  if (now - tStatus >= 1000) {
    uint32_t dt = now - tStatus; tStatus = now;
    micOK = (ringTotal != lastRingSeen); lastRingSeen = ringTotal;
    pps = (pktCount - lastPktCount) * 1000.0f / dt; lastPktCount = pktCount;
    sendStatus(); sendNet();
  }
  if (now - tRes >= 2000) {
    uint32_t dt = now - tRes; tRes = now;
    cpuPct = busyUs / 10.0f / dt;
    busyUs = 0;
    sendResources();
  }
  delay(2);
}