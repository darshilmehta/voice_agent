# Demo script

A 15-minute walk through the MVP: a voice conversation with your documents in English and Hindi, interruptions, live charts, revisiting the chat, a noisy room, and (optionally) live web search. Expected answers are given so you can check them; the documents are fictional and generated (`scripts/eval/build_corpus.py`). Wording varies a little between runs (a local 4B model); the figures don't. Every row was last checked on the real stack over the voice connection on 2026-10-10 (synthetic voices, DESIGN §9).

## 1. Start

Prerequisites (once): `scripts/setup/download_models.sh all`, Docker running, Ollama with `qwen3:4b-instruct`.

**On a 16 GB Mac, cap Ollama's prompt cache first.** Without it, llama-server keeps every prompt's state in RAM (up to 8 GB): in the last check its footprint reached 8.3 GB and swap grew from 2.3 to 12.6 GB in about an hour of conversation, slowing everything (DESIGN §8). Lasts until reboot; DESIGN §9 shows how to make it permanent. If it isn't set, the backend logs a warning and the app shows a notice with this command:

```bash
launchctl setenv LLAMA_ARG_CACHE_RAM 1024 && brew services restart ollama
```

```bash
docker compose -f infra/docker-compose.yml up -d qdrant
```

```bash
cd backend && uv sync --group ml && uv run python -m app
```

```bash
cd frontend && npm install && npm run dev
```

Wait until http://localhost:8000/health shows `preload: ready` (~1–1.5 min cold; Kokoro alone takes ~45–85 s), then open **http://localhost:3000 in Chrome** and allow the microphone.

## 2. The documents

Create a project and upload the five files from `data/eval/docs/` (build them with `uv run scripts/eval/build_corpus.py` from the repo root): `valmora_annual_report_fy24.pdf` (29 pages), `zephyra_investor_deck_q4fy24.pptx`, `valmora_travel_expense_policy.docx`, `suryodaya_yojana_soochna.docx` (Hindi), `valmora_group_health_policy_scan.pdf` (scanned, OCR). All are READY in 2–4 minutes (less once the models are loaded); the project page then shows an **overview** of four panels built from their tables (Valmora KPIs, Valmora quarterly trend, Zephyra KPIs, Suryodaya seats).

## 3. Talk to it

From Home press **Start a conversation** (or open a chat in the project) and tap the **microphone** ("Listening…"). Speak naturally; the agent answers in one or two short sentences, captions light up as it speaks, sources appear as chips.

| Say | Expect |
|---|---|
| "What was Valmora's revenue in FY24?" | ₹7,365 crore |
| "And the EBITDA margin?" | 21.0%; hover the ⓘ beside the message time: "Understood as: What was Valmora's EBITDA margin in FY24?" |
| "How many employees did Valmora have?" | 9,842 (FY24) |
| "What was Zephyra's EBITDA margin in FY24?" | 12.7%, naming the Zephyra investor deck — not Valmora's 21.0% |
| "What is the dividend per share?" | ₹15.00 per share for FY24 (latest year, named) |
| "What is Valmora's corporate identification number?" | L24119GJ1994PLC023871 (say it in full: a spoken "CIN" is often heard as "sin" and gets "The documents do not cover…") |
| "What's the hotel limit for an L3 employee in a Tier-1 city?" | ₹7,500 (chip: § 5.1 Hotel limits per night) |
| "What's the group health policy number?" | GHI/2024/00418377 (from the scanned PDF) |
| "What is EBITDA?" | a short general explanation; on screen "General knowledge, not from your documents" |
| "What is Valmora's FY25 revenue?" | "The documents don't cover Valmora's FY25 revenue." — labelled "Not in your documents", and listed later in the summary as not answered |
| "What's the USD to INR rate today?" (web search off) | "I can't look up live data such as today's rates, prices or news, so I won't guess a figure." |

**Interrupt it.** Ask "Walk me through Valmora's financial performance in FY24", and while it answers say "No wait, I meant the dividend" → the volume dips within ~0.2 s of your first sound, it stops about a second after you start, and answers the dividend (₹15.00 per share for FY24). Say "okay" or "mm-hmm" → the volume dips and comes back after ~1–1.5 s; it keeps going to the end. Say "stop" → it stops ~1–1.5 s after you start, without the volume coming back first; **Esc** stops at once. The transcript shows what you actually heard ("Interrupted after: …").

**Hindi and Hinglish** (first word ~7 s after you stop: Hindi costs the local model more per word).

| Say | Expect |
|---|---|
| "सूर्योदय योजना में आवेदन की अंतिम तिथि क्या है?" | "…अंतिम तिथि 30 नवंबर 2024 है।" |
| "कुल कितनी सीटें हैं?" | "कुल 18,000 सीटें हैं।" |
| "वालमोरा का एफवाई चौबीस में रेवेन्यू कितना था?" | ₹7,365 करोड़, in Hindi. Whisper often mishears the name and "FY24" here ("वाल्मुरा का एट्वाई चावीज…"); the answer is told to name things as the documents do, so it should say "FY24" and "राजस्व"/"रेवेन्यू" rather than the misheard words (a 4B model still slips now and then) |
| "answer in English please" | re-answers the last question in English, and stays English, also for Hindi questions |
| "हिंदी में बताइए" | re-answers the previous question in Hindi, and stays Hindi, also for English questions |
| "answer in English please" | back to English for the rest of the demo |

**Drift.** "What's the capital of France?" → "The capital of France is Paris." (labelled general knowledge). "Let's go back to the annual report" → "Sure, back to the Valmora annual report. What would you like to know?" (a document you name is the one it goes back to; "back to the report" returns to the last document topic: "…We were talking about the seats…").

## 4. Charts on the canvas

| Say | Expect |
|---|---|
| "Show me Valmora's quarterly revenue and EBITDA for FY23 and FY24" | an 8-quarter line chart appears as the answer starts (Q1 FY23 ₹1,512 cr → Q4 FY24 ₹1,933 cr); the particle field docks; the answer quotes the Q4 FY23 and Q4 FY24 values and ends "It's on screen now." Hover or arrow-key a point: value, page, the source cell ("Page 19 · Cell "1,933""). |
| "make that a bar chart" / "only FY24" / "pin this" | "Done." in about a second; the transcript says "Chart updated" |
| "What's the second bar?" | "The second bar represents Q2 FY24 revenue of ₹1,801 crore and EBITDA of ₹372 crore." |
| "show that as a table" / "isko table mein dikhao" | the same data as a table ("हो गया।" for the Hinglish one, whose title may come in Hindi) |
| "Show Zephyra's revenue by segment" | a donut: Freight Services ₹2,609 cr, Contract Logistics ₹1,896 cr, Digital Services ₹481 cr |
| "remove it" / "हटा दो" | "Done." — the newest chart goes |

Every number on a chart comes from a cell of a document table (or a labelled calculation); "View as table" shows the figures. Fact answers with a headline figure may add a small KPI panel, once: a chart whose figures are already on the canvas isn't added again (the transcript says "Already on the canvas" instead of "Chart added"); "remove it" clears the newest.

## 5. Revisit

Open the transcript panel. **Summary** (English or हिंदी, 15–40 s on this machine): key points with page and § chips, "Not answered from your documents" (the FY25 question), follow-ups, and after new messages "Out of date: covers … [Refresh]". **Export**: the chat's ⋯ menu → Export transcript → Markdown or JSON. **Title**: rename the chat, then ⋯ → Regenerate title asks "Replace your title?". **Type instead** sends typed questions into the same chat.

## 6. A noisy room

In a café or with a TV on, the microphone is cleaned (RNNoise) and only speech near your own level opens a turn: background talk is dropped silently, and nothing is said in reply to it. Café chatter 10 dB below your voice got no reply in the last check, and a question asked over it was answered (₹15.00 for "What is the dividend per share?"), though Whisper mishears more in noise (a misheard name gets an honest "The documents do not cover…", and a Hindi question it heard wrong gets "माफ़ कीजिए, मैं ठीक से सुन नहीं पाया…": say it again).

When the room is louder than that (a TV or a talker about as loud as you), switch on **Hold to talk** under the microphone (remembered in this browser): only what you say while holding **Space** (outside a text field) or the talk button is heard, and pressing it while the agent answers stops the answer at once. If voices sound distorted in steady hiss (a fan, an air conditioner), set `voice.noise.denoise: "off"` in `config/local.config.json` and restart the backend.

## 7. Live web search (optional, off by default)

Only the cleaned search query leaves the machine. Enable: start SearXNG (`docker compose -f infra/docker-compose.yml --profile websearch up -d searxng`), set `tools.web_search.enabled: true` and add `"web_search"` to `strict_offline_exceptions` in `config/local.config.json`, restart the backend. Ask "What's the weather in Mumbai today?" → "Let me look that up." (~1.5–2 s), a "Searching the web…" badge with the query, then an answer citing web chips (W1…, with links). Public engines rate-limit automated traffic, so results vary (a rate question may come back "I couldn't find the current rate in the web results" — it never invents one). Stop SearXNG with `docker compose -f infra/docker-compose.yml --profile websearch rm -sf searxng` (never `down`: it also removes Qdrant).

## Known limits

- First spoken word after you stop: English median ~4.7 s (p90 ~6.9 s) on a 16 GB M4; Hindi ~7 s (p90 ~8.6 s); chart edits, "stop" and the resume line ~1–2.7 s. Measured without the prompt-cache cap and with heavy swap: expect a little better with it. A fact answer takes 5–10 s to say, a chart answer ~20–30 s, a web answer ~25 s.
- Whisper-small still garbles some Hindi words and names (more in noise); the agent maps sound-alike names to the documents' names and asks you to repeat when it couldn't make out the question.
- Synthetic voices (Kokoro); Hindi answers are the weakest area.
- The documents here are fictional; the web search can't find them.
