
import { useState, useEffect } from 'react';
import { Mic, MicOff } from 'lucide-react';
import { Button } from './ui/button';
import { AudioRecorder, convertBlobToBase64 } from '@/utils/audioUtils';
import { transcribeAudio } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface AudioRecordButtonProps {
  onTranscription: (text: string) => void;
}

export const AudioRecordButton = ({ onTranscription }: AudioRecordButtonProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioRecorder] = useState(new AudioRecorder());
  const { toast } = useToast();
  
  useEffect(() => {
    return () => {
      // Clean up recording when component unmounts
      if (isRecording) {
        audioRecorder.stopRecording().catch(console.error);
      }
    };
  }, [isRecording, audioRecorder]);

  const handleRecording = async () => {
    if (!isRecording) {
      try {
        await audioRecorder.startRecording();
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
        setIsRecording(false);
        const audioBlob = await audioRecorder.stopRecording();
        
        toast({
          title: "Processing Audio",
          description: "Transcribing your speech...",
        });
        
        const base64Audio = await convertBlobToBase64(audioBlob);
        const transcript = await transcribeAudio(base64Audio);
        
        if (transcript && transcript.trim()) {
          onTranscription(transcript);
          toast({
            title: "Transcription Complete",
            description: "Your speech has been converted to text",
          });
        } else {
          toast({
            title: "No Speech Detected",
            description: "We couldn't detect any speech in your recording",
            variant: "destructive",
          });
        }
      } catch (error) {
        console.error("Audio processing error:", error);
        toast({
          title: "Error",
          description: "Failed to process audio. Please try again.",
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
