import copy
# Create CustomAugmentatror class and call the mask_entities and hide_entities functions
import logging
logger = logging.getLogger("universal_logger")

class AI_augmentator:
    """
    A class to perform AI-based data augmentation for fill-mask and text generation tasks.
    This class uses a generator function to generate new tokens for masked entities in the payload.
    It can also remove masked entities from the payload based on the keep_mask parameter.
    """

    def __init__(self, generator):
        self.generator = generator
        
    # Define augment function for MASK FILLING task
    def augment_fill_mask(self, data, keep_mask):
        """
        Augment the given data based on the fill mask task
        """
        for item in data:
            payload = item['payload']
            entities = item['entities']
            offset = 0
            
            # Search payload for <mask> enitites 
            if payload.find("<mask>") != -1 or payload.find("[MASK]") != -1:
                logger.debug("Found masked entities in payload. Augmenting...\n")

                # If keep_mask is True, generate new tokens for <mask> entities
                # If keep_mask is False, remove <mask> entities from the payload
                if keep_mask:
                    logger.debug("Generating new tokens for masked entities...\n")
                    # For each masked entity, print the generated words with the score and replace the mask with highest scored word
                    for entity in entities:
                        if entity['word'] == '<mask>' or entity['word'] == '[MASK]':
                            generated_words = self.generator(payload)
                            start = entity['start']
                            end = entity['end']
                            word = payload[start:end]
                            logger.info(f"Entity: '{entity['entity_group']}' | Word: '{word}' at position {start}:{end}")

                            if generated_words and isinstance(generated_words[0], list):                
                                logger.debug(f"Predictions for mask at position {start}-{end}:")
                                for i, prediction in enumerate(generated_words[0]):
                                    logger.debug(f"{i+1}. {prediction['token_str']} ({prediction['score']:.6f})")

                                # Select the token with the highest score
                                best_prediction = max(generated_words[0], key=lambda x: x['score'])
                                new_value = best_prediction['token_str']
                                logger.debug(f"Selected token: {new_value} with score: {best_prediction['score']:.6f}\n")

                            elif isinstance(generated_words, list):
                                print(f"Predictions for mask at position {start}-{end}:")
                                for i, prediction in enumerate(generated_words):
                                    logger.debug(f"{i+1}. {prediction['token_str']} ({prediction['score']:.6f})")

                                # Select the token with the highest score
                                best_prediction = max(generated_words, key=lambda x: x['score'])
                                new_value = best_prediction['token_str']
                                logger.debug(f"Selected token: {new_value} with score: {best_prediction['score']:.6f}\n")
                                
                            else:
                                logger.debug(f"No predictions for mask at position {start}-{end}\n") 
                            
                            if new_value:
                                # Replace the original text with the new value in the payload
                                payload = payload[:start] + new_value + payload[end:]

                                entity['word'] = new_value
                                
                                # Calculate length difference
                                length_diff = len(new_value) - len(word)

                                # Update the entity's end position
                                entity['end'] += length_diff

                                # Update offsets for subsequent entities
                                offset += length_diff
                                for future_entity in entities:
                                    if future_entity['start'] > entity['start']:
                                        future_entity['start'] += length_diff
                                        future_entity['end'] += length_diff
                            else:
                                logger.debug(f"No predictions for mask at position {start}-{end}\n")
                else:
                    logger.debug("Removing masked entities from payload...\n")
                    # Remove <mask> entities from the payload
                    new_entities = []
                    for entity in entities:
                        start = entity['start'] + offset
                        end = entity['end'] + offset

                        if entity['entity_group'] == 'mask':
                            payload = payload[:start] + payload[end:]
                            offset -= end - start
                            
                        else:
                            entity['start'] = start
                            entity['end'] = end
                            new_entities.append(entity)
                    entities = new_entities
                print("Augmented payload: ", payload, "\n\n####################################################################################\n")
            else:
                logger.debug("No masks found in payload.")
                logger.debug("Returning original payload...\n")
            offset = 0
            item['payload'] = payload            
            item['entities'] = entities
        return data

   # Call the right augment function based on the specified task
    def augment(self, data, keep_mask):
        """
        Augment the given data based on the specified task
        Args:
            data: list
                The list of dictionaries containing the payload and entities
        Returns:
            augmented_data: list
                The list of dictionaries containing the augmented payload and entities
        """
        augmented_data = copy.deepcopy(data)
        return self.augment_fill_mask(augmented_data, keep_mask)
        
        
    # Define a function to print augmented entities 
    def print_entities(self, data):
        """
        Print the entities from the given data
        Args:
            data: list
                The list of dictionaries containing the payload and entities to print
        """
        # Lists all entities with start and end positions before augmentation
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
            logger.info("\n")
            logger.info("####################################################################################\n\n")