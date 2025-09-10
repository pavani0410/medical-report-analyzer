import React, { useState, useEffect, useRef } from 'react';
import { Stethoscope, Shield, Brain, Sparkles, Zap, UploadCloud, MessageSquare } from 'lucide-react';
import './index.css'; // This line now imports all your consolidated CSS

export default function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [question, setQuestion] = useState('');
    const [chatMessages, setChatMessages] = useState([]);
    const [isAsking, setIsAsking] = useState(false);
    const chatBoxRef = useRef(null);

    // Add initial welcome message
    useEffect(() => {
        setChatMessages([{ type: 'ai', text: 'Welcome! Upload a medical PDF to get started, or ask a general medical question.' }]);
    }, []);

    // Scroll to bottom of chat box whenever messages change
    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [chatMessages]);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setUploadStatus('');
    };

    const handleUpload = async (event) => {
        event.preventDefault();
        if (!selectedFile) {
            setUploadStatus({ message: 'Please select a PDF file to upload.', type: 'error' });
            return;
        }

        setIsUploading(true);
        setUploadStatus({ message: 'Uploading and processing PDF...', type: 'info' });

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                setUploadStatus({ message: result.message, type: 'success' });
                setChatMessages([{ type: 'ai', text: 'PDF processed! You can now ask questions about the report.' }]);
                setSelectedFile(null); // Clear file input
            } else {
                setUploadStatus({ message: result.error || 'An unknown error occurred during upload.', type: 'error' });
            }
        } catch (error) {
            console.error('Upload error:', error);
            setUploadStatus({ message: 'Network error or server unreachable.', type: 'error' });
        } finally {
            setIsUploading(false);
        }
    };

    const handleSendMessage = async () => {
        if (!question.trim()) return;

        const userMessage = { type: 'user', text: question.trim() };
        setChatMessages((prevMessages) => [...prevMessages, userMessage]);
        setQuestion(''); // Clear input

        setIsAsking(true);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: userMessage.text }),
            });

            const result = await response.json();

            const aiMessage = { type: 'ai', text: result.answer || 'No response from AI.' };
            setChatMessages((prevMessages) => [...prevMessages, aiMessage]);
        } catch (error) {
            console.error('Ask error:', error);
            setChatMessages((prevMessages) => [
                ...prevMessages,
                { type: 'ai', text: 'Error: Could not get a response. Please try again.', isError: true },
            ]);
        } finally {
            setIsAsking(false);
        }
    };

    // Function to format markdown-like bold headings and newlines
    const formatAnswer = (text) => {
        // Convert markdown bolding to strong tags
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert newlines to <br> tags for basic line breaks
        formattedText = formattedText.replace(/\n/g, '<br>');

        // Optional: Basic handling for list items (if Gemini generates them with hyphens)
        // This will create a simple unordered list, adjust if Gemini uses other markers
        formattedText = formattedText.replace(/<br>- (.*?)(<br>|$)/g, '<br><li>$1</li>');
        formattedText = formattedText.replace(/<br><li>(.*?)<\/li><br>/g, '<li>$1</li>'); // Clean up extra br around lists
        if (formattedText.includes('<li>')) {
            // Wrap lists in ul tags if they exist
            formattedText = formattedText.replace(/(<li>.*?<\/li>)/s, '<ul>$1</ul>');
        }

        // Ensure there's no <ul><br><li> or </ul><br>
        formattedText = formattedText.replace(/<ul><br>/g, '<ul>');
        formattedText = formattedText.replace(/<br><\/ul>/g, '</ul>');

        return formattedText;
    };

    return (
        <div className="h-screen w-full bg-gradient-to-br from-slate-900 via-blue-900 to-emerald-900 relative overflow-hidden">
            {/* The customCss block and <style> tag are removed from here.
                All CSS is now in frontend/src/index.css and processed by Tailwind. */}

            {/* Animated Background Elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 rounded-full blur-3xl animate-pulse-custom"></div>
                <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-emerald-400/20 to-green-600/20 rounded-full blur-3xl animate-pulse-custom-delay-1000"></div>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-purple-400/10 to-pink-600/10 rounded-full blur-3xl animate-pulse-custom-delay-500"></div>
            </div>

            {/* Grid Pattern Overlay */}
            <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:50px_50px]"></div>

            {/* Header */}
            <div className="relative backdrop-blur-xl bg-white/10 border-b border-white/20 shadow-2xl z-10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between flex-wrap gap-4">
                        <div className="flex items-center gap-4">
                            <div className="relative p-3 bg-gradient-to-br from-cyan-400 to-emerald-500 rounded-2xl shadow-lg group hover:scale-110 transition-transform duration-300">
                                <Stethoscope className="h-8 w-8 text-white" />
                                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
                                <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-emerald-500 rounded-2xl blur-lg opacity-50 group-hover:opacity-70 transition-opacity"></div>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-300 to-emerald-300 bg-clip-text text-transparent">
                                    MedDoc AI
                                </h1>
                                <p className="text-gray-300 flex items-center gap-2 text-sm">
                                    <Sparkles className="h-4 w-4 text-yellow-400 animate-pulse" />
                                    Next-Gen Medical Document Intelligence
                                </p>
                            </div>
                        </div>
                        <div className="flex items-center gap-4 sm:gap-6 flex-wrap justify-end">
                            <div className="flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-green-500/20 backdrop-blur-sm rounded-full border border-green-400/30 hover:bg-green-500/30 transition-colors text-sm">
                                <Shield className="h-4 w-4 text-green-400" />
                                <span className="text-green-300">HIPAA Secure</span>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-blue-500/20 backdrop-blur-sm rounded-full border border-blue-400/30 hover:bg-blue-500/30 transition-colors text-sm">
                                <Brain className="h-4 w-4 text-blue-400" />
                                <span className="text-blue-300">AI Powered</span>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-purple-500/20 backdrop-blur-sm rounded-full border border-purple-400/30 hover:bg-purple-500/30 transition-colors text-sm">
                                <Zap className="h-4 w-4 text-purple-400 animate-pulse" />
                                <span className="text-purple-300">Real-time</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="relative h-[calc(100vh-100px)] p-6 z-10 flex items-center justify-center">
                <div className="h-full w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* File Upload Section - Integrated Directly */}
                    <div className="flex flex-col p-6 bg-gradient-to-br from-blue-800/30 to-purple-800/30 rounded-3xl shadow-2xl border border-blue-700/50 backdrop-blur-md transform hover:scale-[1.01] transition-transform duration-300 flex-1"> {/* Added flex-1 */}
                        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                            <UploadCloud className="h-7 w-7 text-blue-300" /> Medical Document Analysis
                        </h2>
                        <p className="text-gray-300 mb-6">Upload medical documents for AI-powered analysis and insights</p>

                        <form onSubmit={handleUpload} className="flex flex-col items-center justify-center flex-grow p-8 border-2 border-dashed border-blue-500/50 rounded-xl bg-blue-900/10 hover:bg-blue-900/20 transition-colors cursor-pointer">
                            <UploadCloud className="h-20 w-20 text-blue-400 mb-4" />
                            <p className="text-xl text-blue-300 font-semibold mb-2">Upload Medical Documents</p>
                            <p className="text-sm text-gray-400 text-center mb-6">
                                Lab reports • X-rays • MRI • CT scans • Medical records<br/>
                                PDF, DOC, DOCX, TXT, JPG, PNG, DICOM supported
                            </p>
                            <input
                                type="file"
                                accept=".pdf"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-500 file:text-white hover:file:bg-indigo-600 cursor-pointer mb-6"
                            />
                            <button
                                type="submit"
                                disabled={isUploading || !selectedFile}
                                className="w-full py-3 px-6 bg-green-600 hover:bg-green-700 text-white font-bold rounded-full shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                            >
                                {isUploading ? (
                                    <>
                                        <span className="loader"></span> Processing...
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="h-5 w-5" /> Analyze Medical Document
                                    </>
                                )}
                            </button>
                        </form>

                        {uploadStatus.message && (
                            <div className={`mt-4 text-sm text-center ${
                                uploadStatus.type === 'error' ? 'text-red-400' :
                                uploadStatus.type === 'success' ? 'text-green-400' : 'text-blue-300'
                            }`}>
                                {uploadStatus.message}
                            </div>
                        )}

                        <div className="mt-6 text-sm text-gray-400 space-y-2">
                            <p className="flex items-center gap-2"><Shield className="h-4 w-4 text-green-400" /> HIPAA compliant document processing</p>
                            <p className="flex items-center gap-2"><Brain className="h-4 w-4 text-purple-400" /> AI-powered clinical insights and summaries</p>
                        </div>
                    </div>

                    {/* Chat Interface - Integrated Directly */}
                    <div className="flex flex-col p-6 bg-gradient-to-br from-emerald-800/30 to-cyan-800/30 rounded-3xl shadow-2xl border border-emerald-700/50 backdrop-blur-md transform hover:scale-[1.01] transition-transform duration-300 flex-1 min-h-0"> {/* Added min-h-0 */}
                        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                            <MessageSquare className="h-7 w-7 text-emerald-300" /> Medical AI Assistant
                            <span className="ml-auto text-sm text-green-400 flex items-center gap-1">
                                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div> Online
                            </span>
                        </h2>
                        <div ref={chatBoxRef} className="chat-box-content bg-gray-900 rounded-lg p-4 mb-4 border border-gray-700 flex-grow overflow-y-auto"> {/* Added flex-grow and overflow-y-auto */}
                            {chatMessages.map((msg, index) => (
                                <div
                                    key={index}
                                    className={`message p-3 rounded-lg max-w-[85%] ${
                                        msg.type === 'user'
                                            ? 'bg-indigo-600 text-white self-end'
                                            : msg.isError
                                            ? 'bg-red-800 text-red-100 self-start'
                                            : 'bg-gray-700 text-gray-200 self-start'
                                    }`}
                                    dangerouslySetInnerHTML={{ __html: formatAnswer(msg.text) }}
                                ></div>
                            ))}
                            {isAsking && (
                                <div className="message ai-message self-start text-white">
                                    <span className="loader mr-2"></span> Thinking...
                                </div>
                            )}
                        </div>

                        <div className="flex items-center gap-3 bg-gray-800 rounded-full p-2 border border-gray-700 flex-shrink-0"> {/* Added flex-shrink-0 */}
                            <input
                                type="text"
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && !isAsking && handleSendMessage()}
                                placeholder="Ask a question about the report..."
                                disabled={isAsking}
                                className="flex-grow p-2 bg-transparent text-gray-200 focus:outline-none placeholder-gray-500"
                            />
                            <button
                                onClick={handleSendMessage}
                                disabled={isAsking || !question.trim()}
                                className="p-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <Zap className="h-5 w-5" />
                            </button>
                        </div>
                        <div className="mt-4 text-xs text-gray-400 text-center flex-shrink-0"> {/* Added flex-shrink-0 */}
                            AI-powered medical analysis • Always consult qualified professionals
                        </div>
                    </div>
                </div>
            </div>

            {/* Enhanced Footer */}
            <div className="fixed bottom-0 left-0 right-0 backdrop-blur-xl bg-black/20 border-t border-white/10 z-20">
                <div className="max-w-7xl mx-auto px-6 py-3">
                    <div className="flex justify-between items-center text-xs flex-wrap gap-2">
                        <span className="text-gray-400 flex items-center gap-2">
                            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                            © 2025 MedDoc AI - For Healthcare Professionals Only
                        </span>
                        <span className="text-gray-400 text-right">
                            AI-powered medical analysis • Always consult qualified professionals
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
