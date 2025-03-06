
import { useState } from 'react';
import { Speaker, Volume2, VolumeX } from 'lucide-react';
import { Button } from './ui/button';
import { SpeechSynthesizer } from '@/utils/speechUtils';
import { useToast } from '@/hooks/use-toast';

interface AudioPlayButtonProps {
  text: string;
}

export const AudioPlayButton = ({ text }: AudioPlayButtonProps) => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const { toast } = useToast();
  const speechSynthesizer = SpeechSynthesizer.getInstance();

  const handlePlay = async () => {
    if (isSpeaking) {
      speechSynthesizer.stop();
      setIsSpeaking(false);
      return;
    }

    try {
      setIsSpeaking(true);
      await speechSynthesizer.speak(text);
      setIsSpeaking(false);
    } catch (error) {
      console.error("Text-to-speech error:", error);
      toast({
        title: "Speech Error",
        description: "Failed to play audio response",
        variant: "destructive",
      });
      setIsSpeaking(false);
    }
  };

  return (
    <Button
      type="button"
      size="icon"
      variant="ghost"
      onClick={handlePlay}
      className={`${isSpeaking ? 'text-blue-500' : 'text-gray-500'} p-1 h-6 w-6`}
      title={isSpeaking ? "Stop speaking" : "Listen to response"}
    >
      {isSpeaking ? (
        <Volume2 className="h-4 w-4" />
      ) : (
        <Speaker className="h-4 w-4" />
      )}
    </Button>
  );
};
