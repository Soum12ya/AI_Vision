import json
import logging
from typing import List, Dict

# Make sure you have the openai library installed: pip install openai
from openai import OpenAI

from ..config import OPENAI_API_KEY, OPENAI_MODEL

def group_and_summarize_with_llm(detections: List[Dict], schedule: List[Dict]) -> Dict:
    """
    Uses an LLM to group, count, and describe detected symbols based on a lighting schedule.

    Args:
        detections: A list of all detection dicts from the vision pipeline. 
                    Example: [{'symbol': 'A1E'}, {'symbol': 'W'}, {'symbol': 'A1E'}]
        schedule: A list of dicts representing the lighting schedule from the text extractor.

    Returns:
        A dictionary summarizing the counts and descriptions of the lights.
    """
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Cannot perform LLM grouping.")
        # As a fallback, you could implement the simple Python counter here if needed.
        return {"error": "OPENAI_API_KEY is not configured."}

    if not detections:
        logging.warning("No detections were provided to the LLM grouper.")
        return {}

    # 1. Extract just the list of symbols from the detections. This is our raw data.
    detected_symbols = [d.get('symbol') for d in detections if d.get('symbol')]

    # 2. Engineer a clear and specific prompt for the LLM.
    prompt = f"""
    You are an expert electrical estimator. Your task is to count lighting fixtures from a list of detected symbols and provide their descriptions from a schedule.

    CONTEXT:
    Here is the lighting schedule which acts as your rulebook:
    {json.dumps(schedule, indent=2)}

    DATA:
    Here is the complete list of all the fixture symbols detected on the drawings:
    {json.dumps(detected_symbols)}

    INSTRUCTIONS:
    1. Count the occurrences of each unique symbol in the DATA list.
    2. For each unique symbol, find its corresponding "description" from the CONTEXT (lighting schedule).
    3. If a symbol from the DATA is not in the CONTEXT, ignore it.
    4. Your final output must be ONLY a single, valid JSON object. Do not include any other text, explanations, or markdown formatting.
    5. The JSON object should be a dictionary where each key is the symbol and the value is another dictionary containing the "count" and "description".

    EXAMPLE OUTPUT FORMAT:
    {{
      "A1E": {{ "count": 12, "description": "2x4 LED Emergency Fixture" }},
      "W": {{ "count": 9, "description": "Wall-Mounted Emergency LED" }}
    }}
    """

    # 3. Call the OpenAI API with robust error handling.
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only returns valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Set to 0 for deterministic, factual results
            response_format={"type": "json_object"} # Use JSON mode for reliability
        )
        
        response_text = completion.choices[0].message.content
        
        # 4. Parse the JSON response from the LLM.
        return json.loads(response_text)

    except Exception as e:
        logging.error(f"An error occurred while calling the OpenAI API: {e}")
        return {"error": "Failed to process the request with the LLM."}