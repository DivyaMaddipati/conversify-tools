
import { useState, useEffect } from "react";
import { Mic } from "lucide-react";

interface LiveTranscriptionProps {
  transcript: string;
  isRecording: boolean;
}

export const LiveTranscription = ({ transcript, isRecording }: LiveTranscriptionProps) => {
  const [fadeOut, setFadeOut] = useState(false);

  // Set a timer to fade out the component when recording stops
  useEffect(() => {
    let timer: number;
    if (!isRecording && transcript) {
      timer = window.setTimeout(() => {
        setFadeOut(true);
      }, 3000);
    } else {
      setFadeOut(false);
    }

    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [isRecording, transcript]);

  if (!transcript && !isRecording) return null;

  return (
    <div 
      className={`mt-2 p-3 bg-gray-100 rounded-lg text-sm max-w-full transition-opacity duration-500 ${
        fadeOut ? 'opacity-0' : 'opacity-100'
      } ${isRecording ? 'border-l-4 border-red-500' : ''}`}
    >
      {isRecording && (
        <div className="flex items-center gap-2 mb-1 text-red-500 text-xs">
          <Mic className="h-3 w-3 animate-pulse" />
          <span>Recording...</span>
        </div>
      )}
      <p>{transcript || "Listening..."}</p>
    </div>
  );
};
