📄 PageIndex Vectorless RAG

A Vectorless Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents, index them, and ask questions — without using vector embeddings.

Built with:

Streamlit
PageIndex
Azure OpenAI
🚀 Features
📄 Upload and index PDF documents
🔍 Intelligent page retrieval (no embeddings)
💬 Chat with your documents
📊 View document structure and metadata
📌 Page-level citations + confidence score
🧠 How It Works

Unlike traditional RAG (which uses vector databases), this system:

Parses document structure (TOC, pages, sections)
Uses LLM reasoning to identify relevant pages
Retrieves only those pages
Generates answers with citations

👉 This makes it lighter, explainable, and cost-efficient

📂 Project Structure
pageindex-vectorless-rag/
│
├── app.py              # Streamlit UI :contentReference[oaicite:3]{index=3}
├── rag.py              # Core RAG logic :contentReference[oaicite:4]{index=4}
├── requirements.txt    # Dependencies :contentReference[oaicite:5]{index=5}
├── workspace/          # Indexed documents storage
└── .env                # Azure OpenAI config
⚙️ Setup Instructions
1. Clone the repo
git clone https://github.com/shaik23032001/pageindex-vectorless-rag.git
cd pageindex-vectorless-rag
2. Create virtual environment
python -m venv venv
Activate it:
Windows (PowerShell)
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file:

AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=your_model_name
AZURE_OPENAI_API_VERSION=your_version
5. Run the app
python -m streamlit run app.py
💻 Usage
Upload a PDF file
Click Index Document
Select document from sidebar
Ask questions in chat
🧪 Example Query
What are the key findings in this document?

Output includes:

Answer
Page citations
Confidence score
🔧 Tech Stack
Python
Streamlit
PageIndex
Azure OpenAI
LiteLLM
📌 Key Concept: Vectorless RAG

Traditional RAG:

Uses embeddings + vector DB

This project:

❌ No embeddings
✅ Uses document structure + LLM reasoning
⚠️ Requirements
Python 3.9+
Azure OpenAI access
API credentials
📈 Future Improvements
Multi-document querying
Performance optimization
Support for more file types
UI improvements
🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

📄 License

MIT License