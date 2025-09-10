# Medical Report Analyzer

An AI-powered web application for intelligent analysis and conversational querying of medical reports. This project uses a hybrid AI architecture to transform complex, unstructured medical PDF data into clear, concise, and actionable insights.
![Uploading image.png…]()


## Project Overview
The Medical Report Analyzer is a full-stack application designed to assist in the preliminary assessment of medical reports. It allows users to upload a PDF file and ask questions, receiving detailed and context-aware answers.

The system's core functionality is powered by a custom-trained local Large Language Model (LLM) combined with the Google Gemini API, ensuring that responses are not only accurate and relevant to the document but also well-structured and easy to understand.

## Key Features
- **Hybrid AI Architecture**: Combines a locally hosted, fine-tuned LLM with the Google Gemini API for a powerful, yet efficient, AI pipeline.  
- **Retrieval-Augmented Generation (RAG)**: Extracts key clinical information from PDF documents to ground the LLM's responses, preventing hallucinations and ensuring factual accuracy.  
- **Conversational Interface**: An intuitive and responsive chat interface allows users to ask follow-up questions about the report's content.  
- **Structured Output**: The raw output from the local LLM is passed to Gemini, which formats and summarizes the response into a clear, point-by-point structure for better readability.  
- **Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning, making specialization of a large model feasible on consumer-grade hardware.  
- **Robust File Handling**: Securely processes PDF uploads in memory and includes a retry mechanism for file cleanup, ensuring data privacy and system stability.  
- **Modern Stack**: Built with React, Flask, and Tailwind CSS for a clean, fast, and user-friendly experience.  

## Technology Stack
### Backend
- Python/Flask: Main web framework for the API.  
- Hugging Face Transformers/PEFT (LoRA): For loading and fine-tuning the base LLM.  
- LangChain: Orchestrates the RAG pipeline.  
- FAISS: In-memory vector store for efficient semantic search.  
- PyPDF2, io: For robust PDF text extraction.  
- Google Gemini API: For advanced text formatting and summarization.  
- Requests, Accelerate, Torch: For API communication and model optimization.  

### Frontend
- React.js: JavaScript library for building the user interface.  
- Vite: Modern build tool for a fast development experience.  
- Tailwind CSS: Utility-first CSS framework for responsive design.  
- Lucide React: Icon library for visual elements.  

## Project Structure
