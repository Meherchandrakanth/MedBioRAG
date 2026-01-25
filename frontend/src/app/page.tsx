"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Clipboard, Check, Beaker } from "lucide-react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [taskType, setTaskType] = useState("long-form");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleQuery = async () => {
    if (!query) return;
    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          task_type: taskType,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch answer from MediBioRAG API");
      }

      const data = await response.json();
      setAnswer(data.answer);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 py-6 sm:py-12 px-4 sm:px-6 lg:px-8 font-sans text-slate-900">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8 sm:mb-12">
          <div className="flex justify-center mb-4">
            <div className="bg-white p-1 rounded-2xl shadow-lg shadow-brand/10 ring-4 ring-brand/5 overflow-hidden">
              <img
                src="/logo.png"
                alt="MediBioRAG Logo"
                className="w-12 h-12 sm:w-16 sm:h-16 object-contain"
              />
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight mb-2 uppercase italic">
            MediBio<span className="text-brand">RAG</span>
          </h1>
          <p className="text-base sm:text-lg text-slate-600 font-medium px-4">
            Advanced Biomedical Question Answering System
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-200">
          <div className="p-5 sm:p-8">
            <div className="space-y-6">
              <div>
                <label htmlFor="query" className="block text-xs sm:text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wider">
                  Enter biomedical query
                </label>
                <textarea
                  id="query"
                  placeholder="e.g., How does insulin regulate glucose levels?"
                  className="w-full h-32 sm:h-40 px-4 py-3 bg-slate-50 border border-slate-300 rounded-xl focus:ring-2 focus:ring-brand focus:border-brand outline-none transition-all resize-none text-slate-900 placeholder-slate-400 text-sm sm:text-base border-2"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>

              <div className="flex flex-col gap-6">
                <div>
                  <span className="block text-xs sm:text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wider">Answer Mode</span>
                  <div className="flex bg-slate-100 p-1.5 rounded-xl border border-slate-200 w-full overflow-x-auto no-scrollbar gap-1">
                    {["long-form", "mcq", "yes-no"].map((type) => (
                      <button
                        key={type}
                        onClick={() => setTaskType(type)}
                        className={`flex-1 min-w-[90px] px-3 py-2.5 text-[10px] sm:text-xs font-bold uppercase tracking-wider rounded-lg transition-all whitespace-nowrap ${taskType === type
                          ? "bg-white text-brand shadow-md"
                          : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                          }`}
                      >
                        {type.replace("-", " ")}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={handleQuery}
                  disabled={loading || !query}
                  className="w-full bg-brand hover:brightness-110 disabled:grayscale disabled:opacity-50 text-white font-bold py-4 px-6 rounded-xl shadow-lg shadow-brand/30 transition-all flex items-center justify-center space-x-2 active:scale-[0.98]"
                >
                  <Beaker className="w-5 h-5" />
                  <span>Generate Answer</span>
                </button>
              </div>
            </div>
          </div>

          {(answer || error || loading) && (
            <div className="border-t border-slate-100 bg-slate-50/50 p-5 sm:p-8 min-h-[200px]">
              {loading && !answer && (
                <div className="flex flex-col items-center justify-center space-y-4 py-12">
                  <div className="w-10 h-10 border-4 border-brand border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-slate-500 font-medium animate-pulse text-xs sm:text-sm uppercase tracking-widest tracking-widest">Synthesizing clinical evidence...</p>
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1 uppercase font-bold">Error</div>
                  <p className="text-xs sm:text-sm font-medium">{error}</p>
                </div>
              )}

              {answer && (
                <div className="prose prose-slate max-w-none prose-brand">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-2">
                      <div className="w-2.5 h-2.5 bg-brand rounded-full animate-pulse"></div>
                      <span className="text-[10px] sm:text-xs font-bold text-brand uppercase tracking-[0.2em]">MediBioRAG Analysis</span>
                    </div>
                    <button
                      onClick={handleCopy}
                      className="p-2 hover:bg-white hover:shadow-sm rounded-lg transition-all text-slate-500 flex items-center space-x-2 text-[10px] sm:text-xs font-bold border border-transparent hover:border-slate-200 active:bg-slate-100"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-green-600" /> : <Clipboard className="w-3.5 h-3.5 sm:w-4 sm:h-4" />}
                      <span className="uppercase tracking-widest">{copied ? "Copied" : "Copy"}</span>
                    </button>
                  </div>
                  <div className="bg-white p-5 sm:p-8 rounded-2xl border border-slate-200 shadow-sm leading-relaxed text-slate-800 markdown-content text-sm sm:text-base">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        h3: ({ node, ...props }) => <h3 className="text-base sm:text-lg font-bold text-slate-900 mt-8 mb-4 border-l-4 border-brand pl-4" {...props} />,
                        p: ({ node, ...props }) => <p className="mb-5 text-slate-700 leading-relaxed" {...props} />,
                        ul: ({ node, ...props }) => <ul className="list-disc pl-6 mb-5 space-y-3" {...props} />,
                        li: ({ node, ...props }) => <li className="text-slate-700" {...props} />,
                        hr: ({ node, ...props }) => <hr className="my-10 border-slate-100" {...props} />,
                        strong: ({ node, ...props }) => <strong className="text-brand font-bold" {...props} />,
                      }}
                    >
                      {answer}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <p className="text-center mt-8 text-slate-400 text-[10px] sm:text-xs font-bold uppercase tracking-[0.3em] opacity-60">
          Powered by MediBioRAG Search Framework & GPT-4o
        </p>
      </div>
    </main>
  );
}
