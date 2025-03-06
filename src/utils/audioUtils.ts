
export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private stream: MediaStream | null = null;

  startRecording = async (): Promise<void> => {
    try {
      // First stop any existing recording
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
      }
      
      // Release any existing stream
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
      }
      
      // Request a new stream
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      this.mediaRecorder = new MediaRecorder(this.stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.start();
      console.log("Recording started");
    } catch (error) {
      console.error('Error accessing microphone:', error);
      throw error;
    }
  };

  stopRecording = (): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('No recording in progress'));
        return;
      }

      this.mediaRecorder.onstop = () => {
        try {
          // Release the stream tracks
          if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
          }
          
          if (this.audioChunks.length === 0) {
            reject(new Error('No audio data captured'));
            return;
          }
          
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          console.log("Recording stopped, blob size:", audioBlob.size);
          resolve(audioBlob);
        } catch (error) {
          console.error('Error in stopRecording:', error);
          reject(error);
        }
      };

      // Handle errors during recording
      this.mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        reject(new Error('MediaRecorder error'));
      };

      if (this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
      } else {
        reject(new Error('MediaRecorder already inactive'));
      }
    });
  };
}

export const convertBlobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      resolve(base64String.split(',')[1]);
    };
    reader.onerror = (event) => {
      console.error('FileReader error:', event);
      reject(new Error('Failed to convert audio to base64'));
    };
    reader.readAsDataURL(blob);
  });
};
