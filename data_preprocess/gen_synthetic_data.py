from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker  

import json
from typing import List
from pydantic import BaseModel
from litellm import completion
from utils.file_utils import *
from data_preprocess.prompt_template import prompt_template


class Record(BaseModel):
    question: str
    context: str
    answer: str


class Response(BaseModel):
    generated : List[Record]


def get_QA_LLM(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )

    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            data += delta 
    
    return json.loads(data)


if __name__ == "__main__":
    document_converter = DocumentConverter()
    document = document_converter.convert("data_preprocess/COMPS_1376.pdf").document

    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=document)


    data_chunks = []
    for chunk in chunks:
        contextualized_text = chunker.contextualize(chunk=chunk)
        data_chunks.append(contextualized_text)

    with open("data_chunks.json", "w") as f:
        json.dump(data_chunks, f)

    print("Total chunks ", len(data_chunks))

    processed_data = []
    for idx, contextualized_text in enumerate(data_chunks):    
        print(idx)
        llm_data = get_QA_LLM(contextualized_text)

        QA_dict_list = llm_data["generated"]
        for QA_dict in QA_dict_list:
            QA_dict["context"] = contextualized_text

        processed_data.extend(QA_dict_list)
        if idx % 100 == 0:
            save_to_json(processed_data, "data/context_immigration_data.json")
            processed_data = []
     
    save_to_json(processed_data, "data/context_immigration_data.json")




# example_data = """Nikola Tesla (/ˈnɪkələ ˈtɛslə/;[1] Serbian Cyrillic: Никола Тесла [nǐkola têsla]; 10 July 1856 – 7 January 1943) was a Serbian-American[2][3] engineer, futurist, and inventor. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system.[4]

# Born and raised in the Austrian Empire, Tesla first studied engineering and physics in the 1870s without receiving a degree. He then gained practical experience in the early 1880s working in telephony and at Continental Edison in the new electric power industry. In 1884 he immigrated to the United States, where he became a naturalized citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own. With the help of partners to finance and market his ideas, Tesla set up laboratories and companies in New York to develop a range of electrical and mechanical devices. His AC induction motor and related polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and became the cornerstone of the polyphase system which that company eventually marketed."""
# llm_data = get_QA_LLM(example_data)
# processed_data = []
# processed_data.extend(llm_data["generated"])

# print(processed_data)
