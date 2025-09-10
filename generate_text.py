import os
import json
from enum import Enum

from pydantic import BaseModel
from typing import List

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm import LLM, SamplingParams 
from prompts import PROMPT_GENERATE_QUESTIONS, PROMPT_EXTRACT_SENTENCES, SYSTEM_PROMPT_VERIFY_SELF_EXPLANATORY, PROMPT_VERIFY_SELF_EXPLANATORY

class ReferenceSentences(BaseModel):
    sentences: list[str]

class QuestionAnswer(BaseModel):  # Inherit from BaseModel to make it a Pydantic model
    user: str
    assistant: str

class Conversation(BaseModel):
    conversation: List[QuestionAnswer]  # Use the Pydantic model as the type

class VerifySelfExplanatory(BaseModel):
    is_self_explanatory: bool

llm = None

# Function to create prompt for a given document and figure.
def create_prompt(prompt_template, system_prompt=None, **kwargs):
    user_prompt = prompt_template.format(**kwargs)
    if not system_prompt:
        system_prompt = "You are a helpful assistant with knowledge of robotic and engineering."
    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    return conversation

def verify_self_explanatory(sentences) -> list[str]:
    global llm
    json_schema = VerifySelfExplanatory.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1000, guided_decoding=guided_decoding_params)

    conversations = [create_prompt(PROMPT_VERIFY_SELF_EXPLANATORY, system_prompt=SYSTEM_PROMPT_VERIFY_SELF_EXPLANATORY, sentence=sentence) for sentence in zip(sentences)]

    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    results = []
    for output in outputs:
        parsed_output = json.loads(output.outputs[0].text)
        results.append(bool(parsed_output['is_self_explanatory']))
    
    return results

def generate_question(captions, sentences) -> list[str]:
    global llm
    json_schema = Conversation.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1000, guided_decoding=guided_decoding_params)

    spanned_sentences = []
    spanned_captions = []
    span_lengths = []
    for i, sents in enumerate(sentences):
        if len(sents) > 0:
            spanned_sentences.append("\n".join(sents))
            spanned_captions.append(captions[i])
            span_lengths.append(1)
        else:
            span_lengths.append(0)

    # Generate conversations for each figure.
    conversations = [create_prompt(PROMPT_GENERATE_QUESTIONS, caption=caption, sentences=extracted_sentences) for caption, extracted_sentences in zip(spanned_captions, spanned_sentences)]
    # Get outputs from the LLM.
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    # Print the outputs.
    generated_conversations = []
    starting_index = 0
    for span_length in span_lengths:
        conversation = []
        for i in range(starting_index, starting_index + span_length):
            output = outputs[i]
            try:
                parsed_output = json.loads(output.outputs[0].text)
                conversation.extend(parsed_output['conversation'])
            except Exception as e:
                print(f"An unexpected error occurred while parsing extracted sentences: {e}") 

        starting_index += span_length
        generated_conversations.append(conversation)

    return generated_conversations

def verify_sentences(sentences: list, full_text: str):
    verified_sentences = []
    for sentence in sentences:
        # Check if the sentence is in the full text and not belong to the content section.
        if sentence in full_text and "..............................." not in sentence:
            verified_sentences.append(sentence)
    return verified_sentences

def extract_sentences(captions: list, related_texts: list) -> list[str]:
    global llm
    json_schema = ReferenceSentences.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1000, guided_decoding=guided_decoding_params)

    # Generate conversations for each figure.
    spanned_related_texts = []
    spanned_captions = []
    span_lengths = []
    for i, texts in enumerate(related_texts): 
        spanned_related_texts.extend(texts)
        spanned_captions.extend([captions[i]] * len(texts))
        span_lengths.append(len(texts))
        
    conversations = [create_prompt(PROMPT_EXTRACT_SENTENCES, caption=caption, text=text) for caption, text in zip(spanned_captions, spanned_related_texts)]

    # Get outputs from the LLM.
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    extracted_sentences = []
    starting_index = 0
    for span_length in span_lengths:
        sents = []
        for i in range(starting_index, starting_index + span_length):
            output = outputs[i]
            try:
                parsed_output = json.loads(output.outputs[0].text)
                
                verified_sentences = verify_sentences(parsed_output['sentences'], spanned_related_texts[i])
                grouped_sentences = ". ".join(verified_sentences)
                if len(grouped_sentences) > 0: # If there is no extracted sentences, skip it
                    sents.append(grouped_sentences)

                if len(verified_sentences) == 0:
                    print(f"Parsed sentences: {parsed_output['sentences']}")
                    print(f"Texts: {spanned_related_texts[i]}")
            except Exception as e:
                print(f"An unexpected error occurred while parsing extracted sentences: {e}")

        starting_index += span_length
        extracted_sentences.append(sents)
    
    return extracted_sentences

def load_model():
    global llm
    llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4, gpu_memory_utilization=0.98, max_model_len=3000, enforce_eager=True)

def load_data_text(filepath) -> dict:
    with open(filepath, 'r') as file:
        data = json.load(file)
        # Filter items that do not have "texts" or "caption" or have empty values
        filtered_data = [
            item for item in data if 'texts' in item and 'caption' in item and item['texts'] and item['caption']
        ]
        
        return filtered_data

def main():
    output_dir = "gen_output"
    data_text_dir = None
    data_text_filepath = "retrieved_texts.json"
    
    load_model()
    
    filepaths = []
    if data_text_filepath:
        filepaths = [data_text_filepath]
    if data_text_dir:
        input_text_paths = [filename for filename in os.listdir(data_text_dir) if filename.endswith(".json")]
        processed_paths = [filename for filename in os.listdir(output_dir) if filename.endswith(".json")]
        unprocessed_paths = list(set(input_text_paths) - set(processed_paths))
        unprocessed_paths = [os.path.join(data_text_dir, path) for path in unprocessed_paths]
        filepaths.extend(unprocessed_paths)
    
    for file_data in filepaths:
        try:
            print(f"Processing {file_data}")
            # Load the text json file
            text_data = load_data_text(file_data)
            num_samples = len(text_data)

            captions = []
            related_texts = []
            for data in text_data[:num_samples]:
                captions.append(data["caption"])
                related_texts.append(data['texts'])

            extracted_sentences = extract_sentences(captions, related_texts)

            questions = generate_question(captions, extracted_sentences)

            output_data = []
            for data, quests in zip(text_data[:num_samples], questions):
                new_data = {
                    'image_filename': data['image_filename'],
                    'caption': data['caption'],
                    'questions': quests
                }
                output_data.append(new_data)

            output_filename = os.path.basename(file_data)
            output_filename = f'{output_dir}/{output_filename}'
            with open(output_filename, 'w') as file:
                json.dump(output_data, file, indent=4)
            print(f"Saved {output_filename}")
        except Exception as e:
            print(f"An error occurred with file {file_data}:", e)

if __name__ == "__main__":
    main()