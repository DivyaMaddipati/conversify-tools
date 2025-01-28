import { useState, useRef, useEffect } from "react";
import { Send, X, FileText, Image } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { AudioRecordButton } from "./AudioRecordButton";
import { sendChatMessage } from "@/services/api";

interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
  type?: "text" | "image";
  imageUrl?: string;
}

interface ChatInterfaceProps {
  onClose: () => void;
  onResumeMatch: () => void;
}

export const ChatInterface = ({ onClose, onResumeMatch }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isImageMode, setIsImageMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: input.trim(),
      sender: "user" as const,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      if (isImageMode) {
        const response = await fetch("http://localhost:5000/generate-image", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query: input.trim() }),
        });

        if (!response.ok) {
          throw new Error("Failed to generate image");
        }

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        const botMessage = {
          id: Date.now() + 1,
          text: "Here's the image you requested:",
          sender: "bot" as const,
          type: "image" as const,
          imageUrl,
        };

        setMessages((prev) => [...prev, botMessage]);
      } else {
        const data = await sendChatMessage(input.trim());
        const botMessage = {
          id: Date.now() + 1,
          text: data.answer || "Sorry, I couldn't process that request.",
          sender: "bot" as const,
          type: "text" as const,
        };

        setMessages((prev) => [...prev, botMessage]);
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get response from the server",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscription = (text: string) => {
    setInput(text);
    // Create a synthetic event to simulate form submission
    const syntheticEvent = {
      preventDefault: () => {},
    } as React.FormEvent;
    handleSubmit(syntheticEvent);
  };

  const toggleImageMode = () => {
    setIsImageMode(!isImageMode);
    toast({
      title: isImageMode ? "Text Mode" : "Image Mode",
      description: isImageMode 
        ? "Switched to text responses" 
        : "Switched to image generation mode",
    });
  };

  const sampleQuestions = [
    "Contact Information",
    "What about vishnu lms",
    "What about pratibha magazine",
    "What is the email id"
  ];

  return (
    <div className="fixed bottom-4 right-4 w-[680px] h-[600px] bg-white rounded-lg shadow-xl flex flex-col animate-slideIn">
      <div className="p-4 border-b flex justify-between items-center bg-[#0056b3] text-white rounded-t-lg">
        <h2 className="font-semibold">SVECW Chat Assistant</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onResumeMatch}
            className="hover:bg-[#0056b3]/80"
            title="Resume Match"
          >
            <FileText className="h-5 w-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="hover:bg-[#0056b3]/80"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 flex flex-col">
          <ScrollArea className="flex-1 p-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-4 ${
                  message.sender === "user" ? "ml-auto" : "mr-auto"
                }`}
              >
                <div
                  className={`p-3 rounded-lg max-w-[80%] ${
                    message.sender === "user"
                      ? "bg-[#0056b3] text-white ml-auto"
                      : "bg-[#D3E4FD] text-gray-800"
                  }`}
                >
                  {message.text}
                  {message.type === "image" && message.imageUrl && (
                    <img 
                      src={message.imageUrl} 
                      alt="Generated" 
                      className="mt-2 max-w-full rounded-lg"
                    />
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex space-x-2 items-center text-sm text-gray-500">
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce delay-200" />
              </div>
            )}
            <div ref={messagesEndRef} />
          </ScrollArea>

          <form onSubmit={handleSubmit} className="p-4 border-t">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={isImageMode ? "Describe the image you want..." : "Type your message..."}
                disabled={isLoading}
                className="flex-1"
              />
              <AudioRecordButton onTranscription={handleTranscription} />
              <Button
                type="button"
                size="icon"
                onClick={toggleImageMode}
                className={`${
                  isImageMode ? "bg-[#0056b3]" : "bg-gray-200"
                } hover:bg-[#0056b3]/80`}
                title={isImageMode ? "Switch to text mode" : "Switch to image mode"}
              >
                <Image className={`h-4 w-4 ${isImageMode ? "text-white" : "text-gray-600"}`} />
              </Button>
              <Button
                type="submit"
                size="icon"
                disabled={isLoading || !input.trim()}
                className="bg-[#0056b3] hover:bg-[#0056b3]/80"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </form>
        </div>

        <div className="w-64 border-l p-4 bg-gray-50">
          <h3 className="font-semibold mb-3">Sample Questions</h3>
          <div className="space-y-2">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                className="text-sm text-left w-full p-2 hover:bg-[#D3E4FD] rounded transition-colors duration-200"
                onClick={() => setInput(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
