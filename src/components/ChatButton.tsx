import { MessageCircle } from "lucide-react";
import { Button } from "./ui/button";

interface ChatButtonProps {
  onClick: () => void;
}

export const ChatButton = ({ onClick }: ChatButtonProps) => {
  return (
    <Button
      onClick={onClick}
      className="fixed bottom-4 right-4 rounded-full w-12 h-12 shadow-lg hover:shadow-xl transition-shadow duration-300 bg-[#0EA5E9] hover:bg-[#0EA5E9]/80 text-white p-0 flex items-center justify-center"
      aria-label="Open chat"
    >
      <MessageCircle className="w-6 h-6" />
    </Button>
  );
};