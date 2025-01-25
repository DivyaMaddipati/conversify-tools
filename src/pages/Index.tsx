import { useState } from "react";
import { ChatButton } from "@/components/ChatButton";
import { ChatInterface } from "@/components/ChatInterface";
import { ResumeMatch } from "@/components/ResumeMatch";
import bgImage from '@/pages/assets/home.png';

const Index = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isResumeMatchOpen, setIsResumeMatchOpen] = useState(false);

  const handleChatClose = () => {
    setIsChatOpen(false);
    setIsResumeMatchOpen(false);
  };

  return (
    <div 
      className="min-h-screen bg-background relative" 
      style={{
        backgroundImage: `url(${bgImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat'
      }}
    >
      {/* Chat Components */}
      {!isChatOpen && !isResumeMatchOpen && (
        <ChatButton onClick={() => setIsChatOpen(true)} />
      )}
      {isChatOpen && !isResumeMatchOpen && (
        <ChatInterface 
          onClose={handleChatClose}
          onResumeMatch={() => setIsResumeMatchOpen(true)}
        />
      )}
      {isResumeMatchOpen && (
        <ResumeMatch onBack={() => setIsResumeMatchOpen(false)} />
      )}
    </div>
  );
};

export default Index;