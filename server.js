// server.js
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const { PCA } = require('ml-pca');
const { pipeline } = require('@xenova/transformers');

const app = express();
const PORT = process.env.PORT || 3000;
const SECRET = process.env.WEBHOOK_SECRET || '';

const DATA_FILE = path.join(__dirname, 'data', 'sentence.txt');
const VEC_FILE = path.join(__dirname, 'data', 'vectors.json');
const PROJ_FILE = path.join(__dirname, 'data', 'vectors3d.json');

let embedderPromise = null;
async function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embedderPromise;
}

function meanPool(tensorData) {
  const data = tensorData.data || tensorData;
  const dims = tensorData.dims || [1, data.length];
  
  const numTokens = dims[0];
  const embeddingDim = dims[1];
  
  const sum = new Array(embeddingDim).fill(0);
  
  for (let t = 0; t < numTokens; t++) {
    for (let d = 0; d < embeddingDim; d++) {
      sum[d] += data[t * embeddingDim + d];
    }
  }
  
  const mean = sum.map(v => v / numTokens);
  const norm = Math.hypot(...mean) || 1;
  return mean.map(v => v / norm);
}

function loadJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return fallback; }
}
function saveJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

function projectTo3D(vectors) {
  if (!vectors.length) return [];
  const mat = vectors.map(v => v.vec);
  const pca = new PCA(mat, { center: true, scale: false });
  const k = Math.min(3, pca.getEigenvalues().length);
  const proj = pca.predict(mat, { nComponents: k }).to2DArray();
  return vectors.map((row, i) => ({
    step: row.step,
    t: row.t,
    xyz: [proj[i][0] || 0, proj[i][1] || 0, proj[i][2] || 0],
  }));
}

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

  const body = typeof req.body === 'string' ? { sentence: req.body } : (req.body || {});
  const incoming = (body.sentence || '').toString().trim();
  if (!incoming) return res.status(400).json({ error: 'missing sentence' });

  writeSentence(incoming);
  res.set('Content-Type', 'text/plain; charset=utf-8');
  res.send(incoming);

  (async () => {
    try {
      const sentence = readSentence();
      const embed = await getEmbedder();
      const feats = await embed(sentence, { pooling: 'mean', normalize: true });
      const vec = Array.from(feats.data);

      const list = loadJson(VEC_FILE, []);
      const step = (list[list.length - 1]?.step || 0) + 1;
      const row = { step, t: new Date().toISOString(), vec };
      list.push(row);
      saveJson(VEC_FILE, list);

      const proj = projectTo3D(list);
      saveJson(PROJ_FILE, proj);
    } catch (e) {
      console.error('Embedding/PCA error:', e.message, e.stack);
    }
  })();
});

// GET: 3D spiral points
app.get('/vectors3d', (_req, res) => {
  res.set('Content-Type', 'application/json; charset=utf-8');
  res.set('Cache-Control', 'no-store');
  res.json(loadJson(PROJ_FILE, []));
});

// GET: regenerate 3D projections from existing vectors
app.get('/regenerate', (_req, res) => {
  try {
    const list = loadJson(VEC_FILE, []);
    if (!list.length) {
      return res.json({ error: 'no vectors to project' });
    }
    const proj = projectTo3D(list);
    saveJson(PROJ_FILE, proj);
    res.json({ ok: true, points: proj.length });
  } catch (e) {
    res.status(500).json({ error: e.message, stack: e.stack });
  }
});

// POST: reset/clear all data
app.post('/reset', (req, res) => {
  const token = req.header('X-Secret') || '';
  if (!SECRET || token !== SECRET) {
    return res.status(401).json({ error: 'unauthorized' });
  }

  try {
    if (fs.existsSync(DATA_FILE)) fs.unlinkSync(DATA_FILE);
    if (fs.existsSync(VEC_FILE)) fs.unlinkSync(VEC_FILE);
    if (fs.existsSync(PROJ_FILE)) fs.unlinkSync(PROJ_FILE);
    res.json({ ok: true, message: 'All data cleared' });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// health check
app.get('/healthz', (_req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`Rabbit Hole running on http://localhost:${PORT}`);
});
