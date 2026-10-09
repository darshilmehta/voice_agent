/**
 * "A new chat opens straight into voice mode and asks for the microphone" (docs/DESIGN.md §3.9).
 *
 * The click that creates the chat marks it here; the chat page, mounted by the navigation that follows, sees the mark
 * and starts listening. Marking rather than a URL parameter keeps the address clean and means a reload (which has no
 * click behind it, so the browser wouldn't let audio start anyway) never re-triggers it.
 */

const pending = new Set<string>();

export const requestAutoStart = (chatId: string): void => void pending.add(chatId);
export const wantsAutoStart = (chatId: string): boolean => pending.has(chatId);
export const consumeAutoStart = (chatId: string): boolean => pending.delete(chatId);
