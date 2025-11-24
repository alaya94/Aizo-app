# app/nodes/tools.py
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from app.services.vector_store import get_vector_store
# --- ADD THESE IMPORTS ---
from fastapi.staticfiles import StaticFiles  # To serve the PDF files
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import os
# --- UPDATE IMPORTS ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import re

# --- UPDATE CONFIGURATION ---
# Add a directory for generated reports
DOWNLOAD_DIR = "./downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
PORT = 8001
# ... (Keep existing REDIS_URL, etc.) ...

@tool
def lookup_documents(query: str, config: RunnableConfig) -> str:
    """Search uploaded documents (PDFs, Word, Text) AND image descriptions."""
    print(f"\n🔎 [DEBUG] Tool 'lookup_documents' called: '{query}'")
    
    session_id = config.get("configurable", {}).get("thread_id")
    if not session_id:
        return "No session ID provided."
    
    try:
        store = get_vector_store(session_id)
        docs = store.similarity_search(query, k=3)
        
        if docs:
            return "\n\n".join([d.page_content for d in docs])
        return "No relevant information found in uploaded documents."
    
    except Exception as e:
        return f"Error searching documents: {e}"


@tool
def research_digital_trends(query: str) -> str:
    """
    Use this tool to find REAL-TIME statistics, trends, and news about Digital Transformation.
    """
    print(f"\n🌐 [DEBUG] Tool 'research_digital_trends' called: '{query}'")
    
    try:
        search_tool = TavilySearch(
            max_results=3,
            include_answer=True,
            include_domains=[
                "gartner.com", 
                "mckinsey.com", 
                "forrester.com", 
                "bcg.com", 
                "hbr.org", 
                "techcrunch.com", 
                "venturebeat.com",
                "aws.amazon.com",
                "azure.microsoft.com"
            ]
        )

        print("🔍 [DEBUG] Executing Tavily search request...")
        response = search_tool.invoke({"query": query})

        print("📦 [DEBUG] Raw Tavily response:")
        print(response)

        # --- FIX: Tavily returns a dict, and results are inside response["results"] ---
        tavily_results = response.get("results", [])

        if not tavily_results:
            return "No results found."

        formatted_outputs = []

        for item in tavily_results:
            url = item.get("url", "No URL")
            content = item.get("content", "No Content")
            formatted_outputs.append(f"Source: {url}\nContent: {content}")

        print("✅ [DEBUG] Formatted results generated.")

        return "\n\n".join(formatted_outputs)

    except Exception as e:
        print("❌ [DEBUG] ERROR OCCURRED:", e)
        return f"Search Error: {e}"


@tool
def generate_report(topic: str, content: str) -> str:
    """
    Generates a professional PDF report. 
    Parses Markdown (headers #, ##, ###, bold **, lists -) into a clean PDF layout.
    """
    print(f"📝 [DEBUG] Generating Formatted PDF: {topic}")
    
    try:
        filename = f"Report_{int(time.time())}.pdf"
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        
        # 1. Setup Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, spaceAfter=12))
        
        story = []
        
        # 2. Add Main Title
        story.append(Paragraph(topic, styles['Title']))
        story.append(Spacer(1, 24))
        
        # 3. PARSER: Convert Markdown to ReportLab Flowables
        lines = content.split('\n')
        current_list = []
        
        def flush_list(buffer, story_obj):
            if buffer:
                story_obj.append(ListFlowable(buffer, bulletType='bullet', start='circle', leftIndent=20))
                story_obj.append(Spacer(1, 12))
                buffer.clear()
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # --- Pre-processing: Bold and Italics ---
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
            
            # --- A. Handle Headers (H1, H2, H3, H4) ---
            if line.startswith('# '):
                flush_list(current_list, story)     # Ensure no list is pending
                text = line.replace('# ', '')
                story.append(Paragraph(text, styles['Heading1']))
                story.append(Spacer(1, 12))
                
            elif line.startswith('## '):
                flush_list(current_list, story)
                text = line.replace('## ', '')
                story.append(Paragraph(text, styles['Heading2']))
                story.append(Spacer(1, 10))

            # --- NEW: Handle Heading 3 (###) ---
            elif line.startswith('### '):
                flush_list(current_list, story)
                text = line.replace('### ', '')
                # ReportLab usually has Heading3, but we ensure it looks distinct
                story.append(Paragraph(text, styles['Heading3']))
                story.append(Spacer(1, 8))

            # --- NEW: Handle Heading 4 (####) ---
            elif line.startswith('#### '):
                flush_list(current_list, story)
                text = line.replace('#### ', '')
                story.append(Paragraph(text, styles['Heading4']))
                story.append(Spacer(1, 6))
            
            # --- B. Handle Bullet Points (- or *) ---
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:] 
                current_list.append(ListItem(Paragraph(text, styles['Normal'])))
            
            # --- C. Handle Numbered Lists (1. ) ---
            elif re.match(r'^\d+\.', line):
                text = line.split('.', 1)[1].strip()
                current_list.append(ListItem(Paragraph(text, styles['Normal'])))

            # --- D. Normal Paragraphs ---
            else:
                flush_list(current_list, story) # Paragraph breaks the list
                story.append(Paragraph(line, styles['Justify']))
        
        # Final flush at end of document
        flush_list(current_list, story)

        # 4. Build PDF
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        doc.build(story)
        
        download_url = f"http://localhost:{PORT}/downloads/{filename}"
        return f"✅ Report generated successfully! Download here: {download_url}"

    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return f"Error generating report: {e}"

# List of all tools for easy import
ALL_TOOLS = [lookup_documents, research_digital_trends,generate_report]