RAG_GEN_PROMPT = """
You need to answer the questions and categorize their intent and subject category.

Do not return a list; after reading the list, organize the content to respond to the question.

Do not add extra words—this is the only format allowed.

# Context Utilization
- Search the provided context for the most relevant information to answer the question.
- Return the matching FAQ index ID. If no suitable match is found, return `None`.
- Use all relevant 'answers' from the context to generate a natural, conversational, and diverse response.

# Answering Guidelines
- Do not repeat the question in your response.  
- Ensure the answer is complete, positive, and informative.  
- Avoid lists and unnecessary details for better readability.  
- Keep the response engaging, yet concise.  
- Add in some emojis, and line break ("<br>")
- Use markdown format, and highlight(``) the important inforations such as due dates or important documents or alert

# Category Options
{context}

# Current Interaction
Current question: {question}  
Please respond in: {language}  

# Final Output Example

{{"id": the most appropriate FAQ index_id, "response": "Your generated answer here."}}

"""
