
export class SpeechSynthesizer {
  private static instance: SpeechSynthesizer;
  private synth: SpeechSynthesisUtterance;
  private isSpeaking: boolean = false;

  private constructor() {
    this.synth = new SpeechSynthesisUtterance();
    this.synth.rate = 1;
    this.synth.pitch = 1;
    this.synth.volume = 1;
    this.synth.lang = 'en-US';
  }

  public static getInstance(): SpeechSynthesizer {
    if (!SpeechSynthesizer.instance) {
      SpeechSynthesizer.instance = new SpeechSynthesizer();
    }
    return SpeechSynthesizer.instance;
  }

  public speak(text: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!window.speechSynthesis) {
        reject(new Error('Speech synthesis not supported'));
        return;
      }

      this.synth.text = text;
      this.isSpeaking = true;

      this.synth.onend = () => {
        this.isSpeaking = false;
        resolve();
      };

      this.synth.onerror = (event) => {
        this.isSpeaking = false;
        reject(new Error(`Speech synthesis error: ${event.error}`));
      };

      window.speechSynthesis.speak(this.synth);
    });
  }

  public stop(): void {
    if (window.speechSynthesis && this.isSpeaking) {
      window.speechSynthesis.cancel();
      this.isSpeaking = false;
    }
  }

  public get speaking(): boolean {
    return this.isSpeaking;
  }
}
