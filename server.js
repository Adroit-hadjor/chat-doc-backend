import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import dotenv from "dotenv";
import OpenAI from "openai";
import { createRequire } from "module";
import cors from "cors";

import { CloudClient } from "chromadb";

/* 
import { CloudClient } from "chromadb";

const client = new CloudClient({
  apiKey: 'ck-9sJmqLZTk3aZWMyZi3J52bCg9ygR5xPe69mSuLeRavdE',
  tenant: '2173990a-43ae-43df-8fa3-2df15741fc50',
  database: 'DOC_BACKEND'
});
*/

const require = createRequire(import.meta.url);

const rawPdfParse = require("pdf-parse");
const pdfParse = typeof rawPdfParse === "function" ? rawPdfParse : rawPdfParse?.default;

const rawMammoth = require("mammoth");
const mammoth = rawMammoth?.default ?? rawMammoth;

if (typeof pdfParse !== "function") {
  throw new Error("pdf-parse did not resolve to a function. Try `npm i pdf-parse@1`.");
}

dotenv.config();

const app = express();
app.use(bodyParser.json());

// Serve static frontend (public/index.html)
//app.use(cors({ origin: ["http://localhost:8082"], credentials: false }));
app.use(cors());
app.use(express.static("public"));

app.get("/health", (req, res) => res.status(200).send("ok"));


const upload = multer({ storage: multer.memoryStorage() });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- Optional Chroma, with automatic fallback ----------
let chroma = null;
let collection = null;
let useInMemory = false;

// Minimal in-memory vector store
const memStore = []; // { id, text, embedding }

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function topKMemory(query, k = 3) {
  const emb = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: query,
  });
  const q = emb.data[0].embedding;
  return memStore
    .map(s => ({ ...s, score: cosine(q, s.embedding) }))
    .sort((a,b) => b.score - a.score)
    .slice(0, k)
    .map(x => x.text);
}

async function initVectorStore() {
  try {
    if (process.env.CHROMA_URL) {
    chroma = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    // You can omit these if your Cloud account has defaults set:
    tenant: process.env.CHROMA_TENANT,
    database: process.env.CHROMA_DATABASE,
  });
       collection = await chroma.getOrCreateCollection({
  name: process.env.CHROMA_COLLECTION || "docs",
});
      console.log("Connected to Chroma at", process.env.CHROMA_URL);
    } else {
      useInMemory = true;
      console.log("CHROMA_URL not set; using in-memory store.");
    }
  } catch (e) {
    useInMemory = true;
    console.warn("Chroma unavailable; falling back to in-memory store.", e?.message);
  }
}
await initVectorStore();

// ---------- Helpers ----------
function chunkText(text, size = 800, overlap = 120) {
  const chunks = [];
  for (let start = 0; start < text.length; start += Math.max(1, size - overlap)) {
    chunks.push(text.slice(start, start + size));
  }
  return chunks.filter(c => c.trim().length > 0);
}


async function addChunks(chunks) {
  // Embed and store chunks
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const emb = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: c,
    });
    const vec = emb.data[0].embedding;

    if (useInMemory) {
      memStore.push({ id: `chunk-${Date.now()}-${i}`, text: c, embedding: vec });
    } else {
      await collection.add({
        ids: [`chunk-${Date.now()}-${i}`],
        embeddings: [vec],
        documents: [c],
      });
    }
  }
}

async function retrieveTopK(query, k = 3) {
  if (useInMemory) return await topKMemory(query, k);

  // Chroma path
  const qEmb = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: query,
  });
  const results = await collection.query({
    queryEmbeddings: [qEmb.data[0].embedding],
    nResults: k,
  });
  return results.documents?.flat?.() ?? [];
}

// ---------- Upload Route (file picker) ----------
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ message: "No file uploaded." });

    const mime = req.file.mimetype || "";
    let text = "";

    if (mime.includes("pdf")) {
      const parsed = await pdfParse(req.file.buffer);
      text = parsed.text || "";
    } else if (mime.includes("word") || req.file.originalname.endsWith(".docx")) {
      const result = await mammoth.extractRawText({ buffer: req.file.buffer });
      text = result.value || "";
    } else if (mime.includes("text") || req.file.originalname.match(/\.(txt|md)$/i)) {
      text = req.file.buffer.toString("utf8");
    } else {
      return res.status(415).json({ message: "Unsupported file type. Use PDF, DOCX, TXT, or MD." });
    }

    if (!text.trim()) return res.status(400).json({ message: "No extractable text in file." });

    const chunks = chunkText(text);
    await addChunks(chunks);

    res.json({ message: `Uploaded. Indexed ${chunks.length} chunks.` });
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Upload failed.", error: e?.message });
  }
});

// ---------- Ask Route ----------
app.post("/ask", async (req, res) => {
  try {
    const { query } = req.body;
    if (!query?.trim()) return res.status(400).json({ message: "Query is required." });

    const topDocs = await retrieveTopK(query, 4);
    const context = topDocs.join("\n---\n");

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "Answer using only the provided context. If unsure, say you don't know." },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${query}` },
      ],
    });

    res.json({ answer: completion.choices[0].message.content });
  } catch (e) {
    console.error(e);
    res.status(500).json({ message: "Ask failed.", error: e?.message });
  }
});

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`RAG server on http://localhost:${PORT}`));
