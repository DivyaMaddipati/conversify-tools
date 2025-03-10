
import "regenerator-runtime/runtime";
import { useState, useEffect } from "react";
import { Mic, MicOff } from "lucide-react";
import { Button } from "./ui/button";
import SpeechRecognition, { useSpeechRecognition } from "react-speech-recognition";
import { useToast } from "@/hooks/use-toast";

interface AudioRecordButtonProps {
  onTranscription: (text: string) => void;
}

export const AudioRecordButton = ({ onTranscription }: AudioRecordButtonProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const { toast } = useToast();
  
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isMicrophoneAvailable
  } = useSpeechRecognition();

  // Check browser support on component mount
  useEffect(() => {
    if (!browserSupportsSpeechRecognition) {
      console.warn("Browser doesn't support speech recognition.");
    }
  }, [browserSupportsSpeechRecognition]);

  // Reset transcript when recording stops
  useEffect(() => {
    if (!listening && isRecording) {
      setIsRecording(false);
      
      if (transcript) {
        onTranscription(transcript);
        toast({
          title: "Success",
          description: "Speech converted to text",
        });
      }
    }
  }, [listening, isRecording, transcript, onTranscription, toast]);

  const handleRecording = async () => {
    if (!browserSupportsSpeechRecognition) {
      toast({
        title: "Error",
        description: "Your browser doesn't support speech recognition.",
        variant: "destructive",
      });
      return;
    }

    if (!isMicrophoneAvailable) {
      toast({
        title: "Error",
        description: "Microphone is not available. Please check permissions.",
        variant: "destructive",
      });
      return;
    }

    if (!isRecording) {
      try {
        resetTranscript();
        await SpeechRecognition.startListening({ continuous: true, language: "en-US" });
        setIsRecording(true);
        toast({
          title: "Recording Started",
          description: "Speak now...",
        });
      } catch (error) {
        console.error("Microphone access error:", error);
        toast({
          title: "Error",
          description: "Could not access microphone. Please check permissions.",
          variant: "destructive",
        });
      }
    } else {
      try {
        SpeechRecognition.stopListening();
        // The effect hook will handle the transcript processing
      } catch (error) {
        console.error("Recording error:", error);
        toast({
          title: "Error",
          description: "Failed to process speech",
          variant: "destructive",
        });
        setIsRecording(false);
      }
    }
  };

  return (
    <Button
      type="button"
      size="icon"
      onClick={handleRecording}
      className={`${
        isRecording ? "bg-red-500 hover:bg-red-600" : "bg-gray-200 hover:bg-gray-300"
      }`}
      title={isRecording ? "Stop recording" : "Start recording"}
    >
      {isRecording ? (
        <MicOff className="h-4 w-4 text-white" />
      ) : (
        <Mic className="h-4 w-4 text-gray-600" />
      )}
    </Button>
  );
};
