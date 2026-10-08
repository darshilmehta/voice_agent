import { searchKey } from "@/lib/format";

/** Text with the first case-insensitive match of `query` wrapped in <mark>. */
export function Highlight({ text, query }: { text: string; query: string }) {
  if (!query) return <>{text}</>;
  const hay = searchKey(text);
  const needle = searchKey(query);
  const at = hay.indexOf(needle);
  // Lower-casing can change length for a few scripts; skip highlighting rather than mark the wrong span.
  if (at < 0 || hay.length !== text.length) return <>{text}</>;
  return (
    <>
      {text.slice(0, at)}
      <mark>{text.slice(at, at + needle.length)}</mark>
      {text.slice(at + needle.length)}
    </>
  );
}
