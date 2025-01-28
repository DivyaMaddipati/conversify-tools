import { useState } from 'react';
import { Mic } from 'lucide-react';
import { Button } from './ui/button';
import { AudioRecorder, convertBlobToBase64 } from '@/utils/audioUtils';
import { transcribeAudio } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface AudioRecordButtonProps {
  onTranscription: (text: string) => void;
}

export const AudioRecordButton = ({ onTranscription }: AudioRecordButtonProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const { toast } = useToast();
  const audioRecorder = new AudioRecorder();

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
        toast({
          title: "Error",
          description: "Could not access microphone",
          variant: "destructive",
        });
      }
    } else {
      try {
        const audioBlob = await audioRecorder.stopRecording();
        setIsRecording(false);
        
        toast({
          title: "Processing",
          description: "Converting speech to text...",
        });
        
        const base64Audio = await convertBlobToBase64(audioBlob);
        const transcript = await transcribeAudio(base64Audio);
        
        if (transcript) {
          onTranscription(transcript);
          toast({
            title: "Success",
            description: "Speech converted to text",
          });
        }
      } catch (error) {
        console.error('Recording error:', error);
        toast({
          title: "Error",
          description: "Failed to process audio",
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
        isRecording ? "bg-red-500" : "bg-gray-200"
      } hover:bg-opacity-80`}
      title={isRecording ? "Stop recording" : "Start recording"}
    >
      <Mic className={`h-4 w-4 ${isRecording ? "text-white" : "text-gray-600"}`} />
    </Button>
  );
};