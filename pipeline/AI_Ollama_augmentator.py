import re
import logging
logger = logging.getLogger("universal_logger")
from ollama import Client

class AI_ollama_augmentator:
    """
    AI_ollama_augmentator class for generating replacements for masked tokens in log files using the Ollama API.
    This class provides methods to generate replacements for <mask> tokens in log entries,
    augment log data with these replacements, and optionally keep or remove the mask entities.
    """

    def __init__(self, host, model):
        self.client = Client(host=host)
        self.model = model
        
    def _generate_replacements(self, payload: str, initial_temp=0.9) -> list:
        """
        Use Ollama API to generate replacements for <mask>.
        Attempts with decreasing temperature until a valid response is received if needed.
        """

        mask_count = payload.count("<mask>")
        logger.debug(f"Number of <mask> tokens in payload: {mask_count}")
        temperature = initial_temp

        while temperature > 0:
            logger.info(f"Calling Ollama API with temperature={temperature}")
            messages = [
                {
                "role": "system",
                "content": 
                (
                    "You are the world's best professional log file analyst. "
                    "I will provide you with a log where each line contains a <mask> token. "
                    "Your task is to generate suitable replacements for each <mask>. "
                    "Each generated value should be placed on a new line, in the same order as the masks appear. "
                    "Output only the generated values for <mask>, nothing else — no explanations or additional text."
                    "Dont use any words in user prompt."
                )
            },
                {
                    "role": "user",
                    "content": payload
                }
            ]
            
            try:
                stream = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "top_p": 0.9
                    }
                )

                response_text = ""
                for chunk in stream:
                    content = chunk.get("message", {}).get("content", "")
                    response_text += content

                response_text = re.sub(r"\s*<think>.*?</think>\s*", "", response_text, flags=re.DOTALL)
                replacements = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

                if len(replacements) == mask_count:
                    logger.info(f"Generated {len(replacements)} replacements successfully.")
                    return replacements
                else:
                    logger.warning(f"Generated {len(replacements)} replacements, expected {mask_count}.")
                    logger.warning(f"Reducing temperature to {temperature - 0.1}")
                    temperature = round(temperature - 0.1, 1)
            except Exception as e:
                logger.error(f"Error during API call: {e}")
                temperature = round(temperature - 0.1, 1)
        raise ValueError(f"Failed to generate replacements after multiple attempts. Expected {mask_count} replacements, got {len(replacements)}.")
        
    
    def generate_fill_mask(self, data, keep_mask):
        """
        Augment data with filled masks.
        """
        
        logger.debug(f"Keep mask set to: {keep_mask}, augmenting data...\n") if keep_mask else logger.debug("Keep mask set to: False, removing mask entities...\n")
        
        for item in data:
            payload = item['payload']
            entities = item['entities']
            offset = 0

            if "<mask>" in payload or "[MASK]" in payload:
                logger.debug("Found mask entities in payload. Augmenting...\n")
                
                if keep_mask:
                    logger.debug("Generating new tokens for masked entities...")
                    try:
                        generated_words = self._generate_replacements(payload, initial_temp=0.9)
                    except Exception as e:
                        logger.error(f"Generation failed: {e}")
                        continue

                    mask_entities = [e for e in entities if e['word'] in ('<mask>', '[MASK]')]
                    if len(generated_words) != len(mask_entities):
                        logger.warning(f"Mismatch: {len(generated_words)} generated vs {len(mask_entities)} masks.")
                        continue

                    for entity, new_value in zip(mask_entities, generated_words):
                        start = entity['start']
                        end = entity['end']
                        old_word = payload[start:end]
                        old_length = len(old_word)
                        new_length = len(new_value)

                        payload = payload[:start] + new_value + payload[end:]

                        entity['word'] = new_value
                        entity['end'] = start + new_length 

                        length_diff = new_length - old_length
                        offset += length_diff

                        for future_entity in entities:
                            if future_entity is entity:
                                continue
                            if future_entity['start'] > start:
                                future_entity['start'] += length_diff
                                future_entity['end'] += length_diff

                else:
                    logger.debug("Removing masked entities...")
                    new_entities = []
                    for entity in entities:
                        start = entity['start'] + offset
                        end = entity['end'] + offset

                        if entity['entity_group'] == 'mask':
                            payload = payload[:start] + payload[end:]
                            offset -= (end - start)
                        else:
                            entity['start'] = start
                            entity['end'] = end
                            new_entities.append(entity)
                    entities = new_entities
                print("Augmented payload: ", payload, "\n\n####################################################################################\n")

            else:
                logger.debug("No masks found in payload.")

            item['payload'] = payload
            item['entities'] = entities
        return data

    def print_entities(self, data):
        """
        Debug function to print all entities.
        """
        logger.debug("\n[Listing Entities Based on Start-End Positions]:")
        for item in data:
            payload = item['payload']
            logger.info(f"\nPayload: {payload}\nPayload lenght: {len(payload)}")
            for entity in item['entities']:
                entity_group = entity['entity_group']
                start = entity['start']
                end = entity['end']
                word = payload[start:end]
                logger.info(f"Entity: '{entity_group}' | Word: '{word}' | Length: {len(word)} | Start: {start} | End: {end}")
            logger.info("\n####################################################################################\n\n")