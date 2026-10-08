/**
 * A small Server-Sent Events reader for a fetch() response body. EventSource can only GET, and the chat endpoint
 * is a POST, so the stream is parsed here (https://html.spec.whatwg.org/multipage/server-sent-events.html):
 *
 * - bytes are decoded as UTF-8 in streaming mode, so a character split across network chunks (Devanagari is three
 *   bytes per letter) is never garbled;
 * - lines end in LF, CRLF or CR, also when the CR and LF arrive in different chunks;
 * - `event:` names the event, `data:` lines are joined with "\n", one optional space after the colon is dropped,
 *   lines starting with ":" are comments, `id:` and `retry:` are ignored (nothing reconnects);
 * - a blank line dispatches the event. An event left without its blank line when the stream ends is still
 *   delivered (the spec drops it; being lenient costs nothing and keeps a final event a server forgot to close).
 */

export interface SseEvent {
  event: string;
  data: string;
}

export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let event = "";
  let data: string[] = [];
  let seenData = false;

  function* takeLines(final: boolean): Generator<string> {
    let start = 0;
    for (let i = 0; i < buffer.length; i++) {
      const c = buffer[i];
      if (c !== "\n" && c !== "\r") continue;
      // A CR at the very end may be the first half of a CRLF: wait for the next chunk to decide.
      if (c === "\r" && i === buffer.length - 1 && !final) break;
      yield buffer.slice(start, i);
      if (c === "\r" && buffer[i + 1] === "\n") i++;
      start = i + 1;
    }
    buffer = buffer.slice(start);
  }

  function* handle(line: string): Generator<SseEvent> {
    if (line === "") {
      if (seenData) yield { event: event || "message", data: data.join("\n") };
      event = "";
      data = [];
      seenData = false;
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") {
      data.push(value);
      seenData = true;
    }
  }

  let finished = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      for (const line of takeLines(false)) yield* handle(line);
    }
    finished = true;
    buffer += decoder.decode();
    for (const line of takeLines(true)) yield* handle(line);
    if (buffer) yield* handle(buffer);
    buffer = "";
    yield* handle("");
  } finally {
    // The reader stopped early (it returned, or reading failed): let the server know nobody is listening.
    if (!finished) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
