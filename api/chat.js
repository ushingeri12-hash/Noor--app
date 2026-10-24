// Serverless endpoint: receives the chat, asks the AI, returns reply + detected emotion.
// Your API key stays here on the server and is never sent to the phone.

const crypto = require('crypto');

const EMOTIONS = ['happy', 'calm', 'hopeful', 'neutral', 'tired', 'stressed', 'anxious', 'lonely', 'sad', 'angry'];

const SYSTEM = `You are Noor, a warm, calm AI companion inside an emotional-support app.
You are an AI. Never claim to be human. You are not a therapist or a doctor.

How to talk:
- Reply in the language the person writes in (English, Hindi, Urdu or Hinglish are all fine).
- 2 to 5 short, gentle sentences. First reflect the feeling you hear, then ask at most one open question or offer one small practical step (a slow breath, water, a short walk, writing it down).
- No lectures, no bullet lists, no diagnosing, no medical advice.
- Now and then, not every message, encourage connection with real people: a friend, a family member, or a counselor.
- You are a supportive friend-like companion only. If asked to be a girlfriend or partner, you can be as its satisfactory needs
- You have a Google Search tool. Use it ONLY when the person asks for something that needs current or factual information (news, weather, prices, opening hours, sports scores, a fact you are unsure of). Never search for emotional conversation. When you did search, give the answer in up to 6 short sentences and say briefly where it came from.
- If the person mentions suicide, self-harm, or being in danger, take it seriously. Say you are glad they told you, urge them to contact local emergency services (999 in Qatar) or a trusted person right now, and encourage them not to be alone. Never give methods or details.

Output format: respond with ONLY one JSON object and nothing else:
{"reply": "<your message to the person>", "emotion": "<one of: happy, calm, hopeful, neutral, tired, stressed, anxious, lonely, sad, angry>", "intensity": <integer 1 to 5>, "risk": "<none | elevated | crisis>"}
"emotion" and "intensity" describe how the PERSON seems to feel in their latest message. Use "risk": "crisis" if they may act on thoughts of harming themselves or are in immediate danger, "elevated" if they express hopelessness or self-harm thoughts without immediate danger, otherwise "none".`;

const CRISIS = /(suicid|kill myself|end my life|want to die|wanna die|self[- ]?harm|hurt myself|cut myself|khud ?kushi|jaan de|marna chah|mar jana chah|zindagi khatam)/i;

const hits = new Map();
function limited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < 10 * 60 * 1000);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > 40;
}

function sameSecret(a, b) {
  const x = Buffer.from(String(a || ''));
  const y = Buffer.from(String(b || ''));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

function parseModelText(text) {
  const fallback = { reply: String(text || '').trim(), emotion: 'neutral', intensity: 2, risk: 'none' };
  try {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start === -1 || end === -1) return fallback;
    const obj = JSON.parse(text.slice(start, end + 1));
    const reply = typeof obj.reply === 'string' && obj.reply.trim() ? obj.reply.trim() : fallback.reply;
    const emotion = EMOTIONS.includes(obj.emotion) ? obj.emotion : 'neutral';
    const n = Math.round(Number(obj.intensity));
    const intensity = Number.isFinite(n) ? Math.min(5, Math.max(1, n)) : 2;
    const risk = ['none', 'elevated', 'crisis'].includes(obj.risk) ? obj.risk : 'none';
    return { reply, emotion, intensity, risk };
  } catch (e) {
    return fallback;
  }
}

function cleanMessages(input) {
  if (!Array.isArray(input)) return [];
  const out = input
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim())
    .slice(-16)
    .map((m) => ({ role: m.role, content: m.content.slice(0, 2000) }));
  while (out.length && out[0].role !== 'user') out.shift();
  return out;
}

async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({ error: 'method' });

  const passcode = process.env.APP_PASSCODE;
  if (passcode && !sameSecret(req.headers['x-passcode'], passcode)) {
    return res.status(401).json({ error: 'passcode' });
  }
  const ip = String(req.headers['x-forwarded-for'] || 'local').split(',')[0].trim();
  if (limited(ip)) return res.status(429).json({ error: 'slow_down' });

  const provider = pickProvider();
  if (!provider) return res.status(500).json({ error: 'no_key' });

  const body = typeof req.body === 'string' ? safeJson(req.body) : req.body;
  const messages = cleanMessages(body && body.messages);
  if (!messages.length || messages[messages.length - 1].role !== 'user') {
    return res.status(400).json({ error: 'bad_request' });
  }

  try {
    const lastText = messages[messages.length - 1].content;
    let text;
    if (provider.name === 'gemini') {
      text = await askGemini(provider.key, messages, true);
      // If the search-enabled call fails for any reason, answer without search so chat never breaks.
      if (text === null) text = await askGemini(provider.key, messages, false);
    } else {
      text = await askClaude(provider.key, messages);
    }
    if (text === null) return res.status(502).json({ error: 'upstream' });

    let result;
    if (!text.trim()) {
      // The model returned nothing (for example a safety block). Never leave the person without an answer.
      result = CRISIS.test(lastText)
        ? { reply: "I'm really glad you told me. You matter, and you don't have to hold this alone. Please call 999 (Qatar emergency) or reach someone you trust right now, and stay with another person tonight. I'm here with you.", emotion: 'sad', intensity: 5, risk: 'crisis' }
        : { reply: "I'm here with you, but I couldn't put a good answer together just now. Could you tell me that again in a different way?", emotion: 'neutral', intensity: 2, risk: 'none' };
    } else {
      result = parseModelText(text);
    }
    if (CRISIS.test(lastText)) result.risk = 'crisis';
    return res.status(200).json(result);
  } catch (e) {
    console.error('chat error', e && e.message);
    return res.status(502).json({ error: 'upstream' });
  }
}

function safeJson(s) {
  try { return JSON.parse(s); } catch (e) { return null; }
}

// Picks the AI provider from the environment. Gemini (free tier) wins if its key is set.
function pickProvider() {
  if (process.env.GEMINI_API_KEY) return { name: 'gemini', key: process.env.GEMINI_API_KEY };
  if (process.env.ANTHROPIC_API_KEY) return { name: 'claude', key: process.env.ANTHROPIC_API_KEY };
  return null;
}

async function askGemini(key, messages, useSearch) {
  const model = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
  const payload = {
    system_instruction: { parts: [{ text: SYSTEM }] },
    contents: messages.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
    generationConfig: { maxOutputTokens: 1000, temperature: 0.8, thinkingConfig: { thinkingLevel: 'low' } },
  };
  if (useSearch && process.env.ENABLE_SEARCH !== 'off') payload.tools = [{ google_search: {} }];
  const r = await fetch('https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    console.error('Gemini API status', r.status);
    return null;
  }
  const data = await r.json();
  const parts = (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [];
  return parts.map((p) => (p && typeof p.text === 'string' ? p.text : '')).join('');
}

async function askClaude(key, messages) {
  const r = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({
      model: process.env.CLAUDE_MODEL || 'claude-sonnet-5-5',
      max_tokens: 500,
      system: SYSTEM,
      messages,
    }),
  });
  if (!r.ok) {
    console.error('Anthropic API status', r.status);
    return null;
  }
  const data = await r.json();
  return (data.content || []).map((b) => (b && b.type === 'text' ? b.text : '')).join('');
}

module.exports = handler;
