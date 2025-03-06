
import "regenerator-runtime/runtime";
import { useState } from "react";
import { Mic, MicOff } from "lucide-react";
import { Button } from "./ui/button";
import SpeechRecognition, { useSpeechRecognition } from "react-speech-recognition";
import { useToast } from "@/hooks/use-toast";

interface AudioRecordButtonProps {
  onTranscription: (text: string) => void;
}

export const AudioRecordButton = ({ onTranscription }: AudioRecordButtonProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const { transcript, listening, resetTranscript } = useSpeechRecognition();
  const { toast } = useToast();

  const handleRecording = async () => {
    if (!isRecording) {
      try {
        SpeechRecognition.startListening({ continuous: true, language: "en-US" });
        setIsRecording(true);
        resetTranscript();
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
        setIsRecording(false);
        toast({
          title: "Processing",
          description: "Converting speech to text...",
        });

        if (transcript) {
          onTranscription(transcript);
          toast({
            title: "Success",
            description: "Speech converted to text",
          });
        } else {
          toast({
            title: "No Speech Detected",
            description: "We couldn't detect any speech in your recording",
            variant: "destructive",
          });
        }
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
