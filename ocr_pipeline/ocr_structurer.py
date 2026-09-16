import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

# Corrected import
from state import CBCReportJSON

def structure_ocr_to_json(raw_markdown: str) -> dict:
    print("Structuring OCR markdown into JSON...")
    
    # Initialize the fast Groq model
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0, max_tokens=1000)
    
    # Set up the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=CBCReportJSON)
    
    system_prompt = f"""
    You are a precise medical data extraction system.
    Extract the key blood markers from the provided OCR markdown text.
    
    IMPORTANT: Output ONLY valid JSON. Do not include <think> tags or conversational text.
    {parser.get_format_instructions()}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=raw_markdown)
    ]
    
    response = llm.invoke(messages)
    
    # Use your regex cleaner to strip reasoning tags
    clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', response.content, flags=re.DOTALL).strip()
    
    try:
        # Parse the string into a Pydantic object, then dump to a standard Python dictionary
        structured_data = parser.invoke(clean_text)
        return structured_data.model_dump()
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {"error": "Failed to parse OCR data"}