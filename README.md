# Medical Report Analyzer

An AI-powered web application for intelligent analysis and conversational querying of medical reports. This project uses a hybrid AI architecture to transform complex, unstructured medical PDF data into clear, concise, and actionable insights.
<img width="1919" height="899" alt="image" src="https://github.com/user-attachments/assets/2518a2e1-7c31-4226-9058-5def00ce0a17" />


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
.
├── app.py # Flask backend and API endpoints
├── train.py # Script for fine-tuning the LLM with LoRA
├── requirements.txt # Python dependencies
├── frontend/
│ ├── public/
│ ├── src/
│ │ ├── App.jsx # Main React component with UI and logic
│ │ ├── main.jsx # React entry point
│ │ └── index.css # Global and custom CSS (with Tailwind directives)
│ ├── tailwind.config.js # Tailwind CSS configuration
│ ├── postcss.config.js # PostCSS configuration for Tailwind
│ └── vite.config.js # Vite build configuration
├── models/
│ ├── medical_llm_merged_final/ # The base LLM with fine-tuned weights
│ └── adapter_after_extra_training/ # LoRA adapter from training
└── data/
└── medical_qa.jsonl # Fine-tuning dataset

markdown
Copy code

## Installation and Setup
### Prerequisites
- Python 3.8+  
- Node.js & npm (LTS recommended)  
- Git  

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Medical-Report-Analyzer.git
   cd Medical-Report-Analyzer
Create and configure environment variables:
Create a .env file in the project root:

bash
Copy code
GEMINI_API_KEY="your_api_key_here"
Important: Never commit your .env file. Add .env to .gitignore.

Set up the Python backend:

bash
Copy code
python -m venv venv
.\venv\Scripts\activate      # On Windows
pip install -r requirements.txt
Set up the React frontend:

bash
Copy code
cd frontend
npm install
npm run build
Run the application:

bash
Copy code
cd ..
python app.py
Open your browser and visit:

cpp
Copy code
http://127.0.0.1:5000
Disclaimer
This application is for educational and demonstrative purposes only. It is not a substitute for professional medical advice. Always consult a qualified healthcare professional for any health concerns.

License
This project is licensed under the MIT License. See the LICENSE file for details.
