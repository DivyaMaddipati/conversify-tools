
export const transcribeAudio = async (audioBase64: string): Promise<string> => {
  try {
    console.log("Sending audio for transcription, size:", audioBase64.length);
    
    const response = await fetch('http://localhost:5000/transcribe-audio', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ audio: audioBase64 }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Transcription error:', errorData);
      throw new Error(`Failed to transcribe audio: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log("Transcription successful");
    return data.transcript;
  } catch (error) {
    console.error('Error transcribing audio:', error);
    throw error;
  }
};

export const sendChatMessage = async (message: string) => {
  const response = await fetch("http://localhost:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: message }),
  });

  if (!response.ok) {
    throw new Error("Failed to get response");
  }

  return response.json();
};
