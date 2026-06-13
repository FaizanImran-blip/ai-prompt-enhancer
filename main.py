import re
import requests

MODEL_NAME = "qwen2.5:3b-instruct-q4_K_M"


def function_normalization(user_input):
    text = user_input.lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_chunks(user_input):
    chunks = []

    words = user_input.split()

    for word in words:
        chunks.append(word)

    return chunks


def detect_language(user_input, chunks):
    if re.search(r"[\u0600-\u06FF]", user_input):
        return "urdu"

    roman_urdu_words = [
        "btao",
        "batao",
        "kia",
        "kya",
        "kese",
        "kaise",
        "krna",
        "karna",
        "kru",
        "karu",
        "mujhe",
        "muje",
        "bnao",
        "banao",
        "bana",
        "banado",
        "sahi",
        "sae",
        "karo",
        "kro",
        "do",
        "hai",
        "hain",
        "mera",
        "meri",
        "mere",
        "ye",
        "isko",
        "usko",
        "mujko",
        "likho",
        "samjhao",
        "khrab",
        "raha",
        "rahi",
        "hota",
        "hoti",
        "ni",
        "nahi",
        "chal",
        "chalao",
        "kyu",
        "kyun",
        "kaam",
    ]

    roman_count = sum(1 for word in chunks if word in roman_urdu_words)

    if roman_count >= 1:
        return "roman_urdu"

    return "english"


creation_words = [
    "make",
    "create",
    "build",
    "generate",
    "bnao",
    "banao",
    "bana",
    "banado",
    "app",
    "website",
    "ui",
    "screen",
    "dashboard",
]

debug_words = [
    "fix",
    "error",
    "bug",
    "issue",
    "sae",
    "sahi",
    "theek",
    "khrab",
    "not",
    "running",
    "crash",
    "failed",
    "nahi",
    "ni",
    "chal",
    "solve",
]

explain_words = [
    "explain",
    "what",
    "why",
    "btao",
    "batao",
    "samjhao",
    "kia",
    "kya",
    "how",
    "kaise",
    "kese",
    "meaning",
]

writing_words = [
    "write",
    "email",
    "letter",
    "likho",
    "message",
    "reply",
    "application",
    "caption",
    "post",
    "content",
]


def extract_metadata(user_input, chunks):
    metadata = {}

    metadata["word_count"] = len(chunks)
    metadata["language"] = detect_language(user_input, chunks)

    if not chunks:
        metadata["task_type"] = "empty"
        metadata["keywords"] = []
        return metadata

    if any(word in chunks for word in writing_words):
        metadata["task_type"] = "writing"

    elif any(word in chunks for word in debug_words):
        metadata["task_type"] = "debugging"

    elif any(word in chunks for word in explain_words):
        metadata["task_type"] = "explanation"

    elif any(word in chunks for word in creation_words):
        metadata["task_type"] = "creation"

    else:
        metadata["task_type"] = "general"

    stop_words = {
        "the",
        "is",
        "a",
        "an",
        "to",
        "of",
        "and",
        "or",
        "me",
        "my",
        "i",
        "this",
        "that",
        "in",
        "on",
        "for",
        "with",
        "ko",
        "ke",
        "ka",
        "ki",
        "se",
        "do",
        "hai",
        "hain",
        "ho",
        "ye",
        "mujhe",
        "muje",
        "ai",
    }

    metadata["keywords"] = [
        word for word in chunks if word not in stop_words and len(word) > 2
    ]

    return metadata


def build_prompt(user_input, metadata):
    return f"""
Enhance this user prompt.

Original prompt:
{user_input}

Metadata:
{metadata}

Make it clear, detailed, and professional.
Do not change the original meaning.
Return only enhanced prompt.
"""


def function_response(model_prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL_NAME, "prompt": model_prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Ollama server is not running. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "Error: Model response timed out."
    except Exception as e:
        return f"Error: {e}"


def main():
    while True:
        user_input = input("Enter your prompt: ")

        normalization = function_normalization(user_input)
        chunks = create_chunks(normalization)
        metadata = extract_metadata(user_input, chunks)
        model_prompt = build_prompt(user_input, metadata)
        ai_response = function_response(model_prompt)

        print("\nUser Input:")
        print(user_input)
        print("\nNormalization Done:")
        print(normalization)
        print("\nChunks List:")
        print(chunks)
        print("\nMetadata:")
        print(metadata)
        print("\nModel Prompt:")
        print(model_prompt)
        print("\nAI Enhanced Prompt:")
        print(ai_response)


if __name__ == "__main__":
    main()
