// server.js
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;
const SECRET = process.env.WEBHOOK_SECRET || '';

const DATA_FILE = path.join(__dirname, 'data', 'sentence.txt');

// helpers
function readSentence() {
  try { return fs.readFileSync(DATA_FILE, 'utf8').trim(); }
  catch { return ''; }
}
function writeSentence(text) {
  fs.writeFileSync(DATA_FILE, text, 'utf8');
}

// middleware
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '100kb' }));     // accept JSON bodies
app.use(express.text({ type: 'text/plain' }));  // also accept plain text
app.use(express.static(path.join(__dirname, 'public')));

// GET: current sentence (plain text)
app.get('/sentence', (_req, res) => {
  res.set('Content-Type', 'text/plain; charset=utf-8');
  res.set('Cache-Control', 'no-store');
  res.send(readSentence());
});

// POST: webhook from n8n with full updated sentence
app.post('/webhook/update', (req, res) => {
  const token = req.header('X-Secret') || '';
  if (!SECRET || token !== SECRET) {
    return res.status(401).json({ error: 'unauthorized' });
  }

  // accept either JSON {sentence:"..."} or raw text
  const body = typeof req.body === 'string' ? { sentence: req.body } : (req.body || {});
  const incoming = (body.sentence || '').toString().trim();
  if (!incoming) return res.status(400).json({ error: 'missing sentence' });

  // enforce +1 word rule (optional but recommended)
  const prev = readSentence();
  const count = s => s.trim().split(/\s+/).filter(Boolean).length;
  if (prev && count(incoming) !== count(prev) + 1) {
    return res.status(400).json({ error: 'must append exactly one word' });
  }
  if (/[.!?]$/.test(incoming)) {
    return res.status(400).json({ error: 'do not end the sentence with punctuation' });
  }

  writeSentence(incoming);
  res.set('Content-Type', 'text/plain; charset=utf-8');
  res.send(incoming);
});

// health check
app.get('/healthz', (_req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`Rabbit Hole running on http://localhost:${PORT}`);
});
