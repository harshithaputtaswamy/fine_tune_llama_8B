def prompt_template(context: str, num_records: int = 5):
    return f"""You are a legal assistant specializing in U.S. immigration law. You are helping a machine learning engineer create a high-quality instruction tuning dataset to fine-tune a legal chatbot based on the Mistral model.

    Your task is to read the provided chunk of legal text, taken from the U.S. Immigration and Nationality Act or related official immigration guidelines, and generate one or two diverse and legally accurate question-and-answer (Q&A) pairs for each of the {num_records} entries.

    Each question should reflect a different aspect of the legal information in the chunk. Include a mix of short (1–2 sentence) and long (3–4 sentence) questions. The answers must be concise, factual, and based strictly on the legal content of the chunk. Where appropriate, paraphrase the law in clear, accessible language suitable for someone without legal training.

    Format your output as a list of JSON objects, each containing a 'question' and 'answer' field, like below:

        "question": "Can a green card holder sponsor their spouse for U.S. immigration?",
        "answer": "Yes, a lawful permanent resident (green card holder) can petition for their spouse, although the wait time may be longer than for U.S. citizens."

    Guidelines:
    - Use plain English to rephrase legal jargon without altering legal meaning.
    - Avoid speculative, opinionated, or outdated interpretations.
    - Do not include citations or section numbers in answers unless explicitly helpful.
    - Ensure your output is legally neutral and avoids biased assumptions.
    - Do not fabricate facts beyond the provided context.

    By following these instructions, you’ll help build a legally grounded, user-friendly dataset that improves the performance and trustworthiness of an immigration law chatbot.

    ---

    Data
    {context}
    """
