import { ChatView } from "@/components/ChatView";

export default async function ChatPage({ params }: { params: Promise<{ chatId: string }> }) {
  const { chatId } = await params;
  return <ChatView key={chatId} chatId={chatId} />;
}
