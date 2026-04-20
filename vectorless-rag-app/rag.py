import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from PageIndex.pageindex import PageIndexClient, utils

WORKSPACE_DIR = "./workspace"

def configure_azure_openai():
    """Set Azure OpenAI environment variables from .env file."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    missing = [
        name for name, value in [
            ("AZURE_OPENAI_ENDPOINT", azure_endpoint),
            ("AZURE_OPENAI_API_KEY", azure_api_key),
            ("AZURE_OPENAI_DEPLOYMENT", azure_deployment),
            ("AZURE_OPENAI_API_VERSION", azure_api_version),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Azure OpenAI environment variables are not fully set in .env file. "
            f"Missing: {', '.join(missing)}"
        )
    os.environ["AZURE_API_KEY"] = azure_api_key
    os.environ["AZURE_API_BASE"] = azure_endpoint
    os.environ["AZURE_API_VERSION"] = azure_api_version

    return f"azure/{azure_deployment}"

class VectorlessRAG:
    def __init__(self, workspace_dir: str = WORKSPACE_DIR):
        self.model = configure_azure_openai()
        self.client = PageIndexClient(model=self.model, retrieve_model=self.model, workspace=workspace_dir)
        self.retrieve_model = self.client.retrieve_model
        self.workspace = Path(workspace_dir)
        print(f"workspace: {self.workspace.absolute()}")

    def index_document(self, pdf_path: str):
        """Index a document at the given path."""
        pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"Document not found at path: {pdf_path}")
        print(f"Indexing document: {pdf_path}")
        print(" this may  take a few moments...")

        pdf_name= Path(pdf_path).name
        for doc_id, doc in self.client.documents.items():
            if doc.get('doc_name') == pdf_name:
                print(f"Document {pdf_name} is already indexed with doc_id: {doc_id}")
                return doc_id
        doc_id = self.client.index(pdf_path)

        print(f"Document indexed with doc_id: {doc_id}")
    
        return doc_id
    
    def get_document_info(self, doc_id: str) -> dict:
        """Get metadata for a document by its ID."""
        return json.loads(self.client.get_document(doc_id))
    def get_structure(self, doc_id: str) -> list:
        """Get the structure of a document (e.g. table of contents)."""
        return json.loads(self.client.get_document_structure(doc_id))
    def print_structure(self, doc_id: str):
        """Print the document structure in a readable format."""
        structure = self.get_structure(doc_id)
        utils.print_structure(structure)
    
    def query(self, doc_id: str, question: str, verbose: bool = True) -> dict:
        """Query a document by its ID and a natural language question."""
        if verbose:
            print(f"Question: {question}")
            print("_" * 60)
        structure = self.get_structure(doc_id)
        doc_info = self.get_document_info(doc_id)

        if verbose:
            print("Document structure:")
            print(f"Document: {doc_info.get('doc_name', 'Unknown')}")
            print(f"Pages: {doc_info.get('page_count', 'unknown')}")

        if verbose:
            print("Retrieving relevant content...")
        relevant_pages = self._find_relevant_pages(structure, doc_info,question)
        if verbose:
            print(f"Identified relevant pages: {relevant_pages}")
        
        if verbose:
            print("\n Retrieving page content...")

        content = self.client.get_page_content(doc_id, relevant_pages)
        content_data = json.loads(content)
        if verbose:
            print("Generating Answers...")
        answer = self._generate_answer(question, content_data, doc_info)
        answer['pages_searched'] = relevant_pages

        if verbose:
            print("\nAnswer:")
            print(answer['answer'])
            print("\nSources:")
            
        return answer
    def _find_relevant_pages(self, structure: list, doc_info: dict, question: str) -> str:
        from PageIndex.pageindex.utils import llm_completion
        prompt = f"""You are a document nnavigation expert. Your task is to identify which pages 
of a document are most likely to contain the answer to a question.
DOCUMENT: {doc_info.get('doc_name', 'Unknown')}
Description: {doc_info.get('doc_description', 'No description')}
TOTAL PAGES: {doc_info.get('page_count', 'unknown')}
DOCUMENT STRUCTURE (hierarchial table of contents with summeries):
{json.dumps(structure, indent=2)[:10000]}

USER QUESTION: {question}
Based onn the section titles, summeries and page ranges (start_index to end_index), identify which pages are most likely to contain the answer.

Think step by step:
1. What is the user asking about?
2. Which sections' titles/summeries mention related topics?
3. What are the page ranges for those sections?

return a JSON object:
{{
    "reasoning": "your step-by-step reasoning about why these pages are relevant."
    "relevant_sections": ["section title 1", "section title 2"],
    "pages": "5-7, 12-15 //Forma: comma-seperated ranges or single pages"

}}

Important:
-select focused page ranges, not the entire document
-use format like '5-7 for ranges or '12' for ranges "3,8,12" for individual pages
-maximum 15-20 pages total.

Return only the JSON object, no additional text."""
        
        response = llm_completion(model=self.retrieve_model, prompt=prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response.strip())
            return result.get('pages', '1-5')
        except:
            return '1-5'
    def _generate_answer(self, question: str, content_data: list, doc_info: dict) -> dict:
        from PageIndex.pageindex.utils import llm_completion
        content_text= ""
        pages_retrieved = []
        for item in content_data:
           if isinstance(item, dict):
               page_num = item.get('page', 'unknown')
               pages_retrieved.append(page_num)
               content_text += f"\n\n--- Page {page_num} ---\n\n{item.get('content', '')}\n"
        prompt = f"""You are an expert at answering questions based on document content.

DOCUMENT: {doc_info.get('doc_name', 'Unknown')}

RETRIVED CONTENT:
{content_text[:20000]}
USER QUESTION: {question}

Provide a clear, acurate answer based ONLY on the information in the retrieved content. 
Include page citations where you found the information.
Return a JSON object:
{{
    "answer": "your answer to the user's question based on the retrieved content.",
    "citations": [list of pagge numbers where you found the information],
    "confidence": "high/medium/low",
    "key_points": ["point 1", "point 2"]
}}

If the answer cannot be found in the provided content ,say so cearly and set confidence to "low".
Return ONLY the JSON object, no additional text."""
        response = llm_completion(model=self.model, prompt=prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response.strip())
            return result
        except:
            return {
                "answer": "Sorry, I couldn't generate an answer based on the retrieved content.",
                "citations": pages_retrieved,
                "confidence": "Unkonown"
            }
    def list_documents(self) -> list:
        """List all indexed documents with their metadata."""
        docs=[]
        for doc_id in self.client.documents:
            docs.append(self.get_document_info(doc_id))
        return docs
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vectorless RAG System")
    parser.add_argument('--index', type=str, help="Path to a PDF file to index")
    parser.add_argument('--query', type=str, help="Question to ask")
    parser.add_argument('--doc_id', type=str, help="Document ID to query")
    parser.add_argument("--list", action="store_true", help="List all indexed documents")
    args = parser.parse_args()

    rag = VectorlessRAG()

    if args.list:
        print("Indexed Documents:")
        for doc in rag.list_documents():
            print(f"- {doc.get('doc_name', 'Unknown')} (ID: {doc.get('doc_id', 'N/A')[:8]}...)")
        return
    if args.index:
        doc_id = rag.index_document(args.index)
        print(f"Document indexed with ID: {doc_id}")
    
    if args.query and args.doc_id:
        result = rag.query(args.doc_id, args.query)
        return 
    
    print("\n" + "=" *60)
    print("vectorless RAG System - Interactive mode")
    print("=" *60)

    docs = rag.list_documents()
    if not docs:
        print("No documents indexed yet. Use --index to add a document.")
        return
    print("\n Available Documents:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.get('doc_name', 'Unknown')} (ID: {doc.get('doc_id', 'N/A')[:8]}...)")

    while True:
        try:
            choice = input ("\n select document number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return 
            idx = int(choice) - 1
            if idx < 0 or idx >= len(docs):
                doc_id = list(rag.client.documents.keys())[idx]
                break
        except ValueError:
            pass
        print("Invalid input. Try again.")
    print(f"Selected document: {docs[idx].get('doc_name')}")
    print("Enter your question (or 'q' to quit):")

    while True:
        try:
            quesstion = input("Question:").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ['q', 'quit', 'exit', 'q']:
            break
        if not question:
            print("Please enter a question.")
            continue
        rag.query(doc_id, question)


if __name__ == "__main__":
    main()
    
