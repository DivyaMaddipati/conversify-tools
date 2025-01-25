import { useState } from "react";
import { Upload, ArrowLeft, Percent } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { useToast } from "@/hooks/use-toast";

interface ResumeMatchProps {
  onBack: () => void;
}

export const ResumeMatch = ({ onBack }: ResumeMatchProps) => {
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setResume(file);
    } else {
      toast({
        title: "Invalid file",
        description: "Please upload a PDF file",
        variant: "destructive",
      });
    }
  };

  const handleSubmit = async () => {
    if (!resume || !jobDescription.trim()) {
      toast({
        title: "Missing information",
        description: "Please provide both resume and job description",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("resume", resume);
    formData.append("jobDescription", jobDescription);

    try {
      const response = await fetch("http://localhost:5000/match", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to process");

      const data = await response.json();
      setSimilarity(data.similarity);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to process the matching request",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-4 right-4 w-[680px] h-[600px] bg-white rounded-lg shadow-xl flex flex-col animate-slideIn">
      <div className="p-4 border-b flex justify-between items-center bg-primary text-white rounded-t-lg">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onBack}
            className="hover:bg-primary-foreground/10"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h2 className="font-semibold">Resume Match</h2>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        <div className="space-y-6">
          {/* Resume Upload Section */}
          <div className="space-y-2">
            <label className="block text-sm font-medium">Upload Resume (PDF)</label>
            <div className="border-2 border-dashed rounded-lg p-6 text-center">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
                id="resume-upload"
              />
              <label
                htmlFor="resume-upload"
                className="cursor-pointer flex flex-col items-center gap-2"
              >
                <Upload className="h-8 w-8 text-gray-400" />
                <span className="text-sm text-gray-600">
                  {resume ? resume.name : "Click to upload or drag and drop"}
                </span>
              </label>
            </div>
          </div>

          {/* Job Description Section */}
          <div className="space-y-2">
            <label className="block text-sm font-medium">Job Description</label>
            <Textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job description here..."
              className="h-32"
            />
          </div>

          {/* Submit Button */}
          <Button
            onClick={handleSubmit}
            disabled={isLoading || !resume || !jobDescription.trim()}
            className="w-full"
          >
            {isLoading ? "Processing..." : "Calculate Match"}
          </Button>

          {/* Results Section */}
          {similarity !== null && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Match Score:</span>
                <div className="flex items-center gap-2">
                  <Percent className="h-4 w-4 text-primary" />
                  <span className="text-lg font-bold text-primary">
                    {Math.round(similarity)}%
                  </span>
                </div>
              </div>
              <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-500 ease-out"
                  style={{ width: `${similarity}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};