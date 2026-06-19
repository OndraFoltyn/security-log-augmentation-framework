from pm_deep_augmentation.deep_augmentator_service.AI_augmentator import AI_augmentator
from pm_deep_augmentation.deep_augmentator_service.AI_Ollama_augmentator import AI_ollama_augmentator
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import torch, json
import logging
logger = logging.getLogger("universal_logger")
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.artifacts
import argparse


# Main class for Deep Augmentation using MLflow model
class DeepAugmentator:
    def __init__(self, model_path, tokenizer_path, keep_mask: bool):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = self._initialize_cuda()
        self.augmentator = self._initialize_augmentator(keep_mask)

    def _initialize_cuda(self):
        """
        Check if CUDA is available and set the device.
        """
        logger.debug(f"### Scanning for available device. ###")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"### Using: {device} device ###\n\n")
        return device
        
    def _initialize_augmentator(self, keep_mask):
        """
        Initialize the AI_augmentator based on whether keep_mask is True or False.
        If keep_mask is True, load the model and tokenizer; otherwise, return a placeholder.
        """
        if keep_mask:
            # Load model, tokenizer, and pipeline once
            task = "fill-mask"
            model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            model.to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            pipe = pipeline(task, model=model, tokenizer=tokenizer, device=0 if self.device.type == 'cuda' else -1)
            return AI_augmentator(pipe)
        else:
            # Model and tokenizer are not needed if keep_mask is False
            logger.info("Model and tokenizer are not needed if keep_mask is False.")
            return AI_augmentator(None)
    
    def deep_augmentate(self, data, keep_mask, print_entities):
        """
        Main augmentate method.
        """
        filled_data = self.augmentator.augment(data, keep_mask)
        if print_entities:
            self.augmentator.print_entities(filled_data)
        return filled_data


# Main class for Ollama API-based augmentation
class DeepOllamaAugmentator:
    def __init__(self, host, model_name, keep_mask: bool):
        self.host = host
        self.model_name = model_name
        self.augmentator = self._initialize_augmentator(keep_mask)
        
    def _initialize_augmentator(self, keep_mask):
        """
        Initialize the AI_ollama_augmentator based on whether keep_mask is True or False.
        If keep_mask is True, initialize the Ollama augmentator; otherwise, return a placeholder.
        """
        if keep_mask:
            logger.debug(f"### Initializing Ollama augmentator with model: {self.model_name} ###")
            print(f"### Initializing Ollama augmentator with model: {self.model_name} ###")
            return AI_ollama_augmentator(host=self.host, model=self.model_name)
        else:
            logger.info("Ollama model is not needed if keep_mask is False.")
            print("Ollama model is not needed if keep_mask is False.")
            return AI_ollama_augmentator(None, None)

    def deep_augmentate(self, data, keep_mask, print_entities):  
        """
        Main augmentate method.
        """
        generated_data = self.augmentator.generate_fill_mask(data, keep_mask)
        self.augmentator.print_entities(generated_data) if print_entities else None
        return generated_data



# Call main
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for mask replacement using either API or MLM model in MLflow.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose the augmentation mode")

    # API-based augmentation
    api_parser = subparsers.add_parser("api", help="Use Ollama API generator")                                                                                                  
    api_parser.add_argument("--API_host", type=str, required=True, help="Ollama server host. Nebula: http://192.168.40.2:11434. Aurora: http://192.168.40.5:11434")
    api_parser.add_argument("--API_model", type=str, required=True, help="Ollama model name")

    # Deep augmentation
    deep_parser = subparsers.add_parser("deep", help="Use Deep Augmentator (MLflow model)")
    deep_parser.add_argument("--deep_model_name", type=str, required=True, help="MLflow model name")
    deep_parser.add_argument("--deep_model_alias", type=str, required=True, help="MLflow model alias")

    parser.add_argument("--print_entities", action="store_true", help="Print entities in the augmented data")
    parser.add_argument("--keep_mask", action="store_true", help="Keep the <mask> in the augmented data")
    args = parser.parse_args() 
    
    # Example of cmd line arguments:
    # python pipeline.py --print_entities --keep_mask deep --deep_model_name "Roberta-base" --deep_model_alias "newest" 
    # python pipeline.py --print_entities --keep_mask api --API_host "http://192.168.40.5:11434" --API_model "llama3:70b" 

    # Data input for testing
    data_input = [
        {"entities": [
            {"entity_group": "rule", "start": 74, "end": 96, "word": "traffic:forward accept"}, 
            {"entity_group": "os_name", "start": 994, "end": 1011, "word": "Windows 10 / 2016"}, 
            {"entity_group": "host_name", "start": 286, "end": 301, "word": "WIN-7OCV2UEAUKA"}, 
            {"entity_group": "event_type", "start": 161, "end": 176, "word": "traffic:forward"}, 
            {"entity_group": "device_ip", "start": 21, "end": 35, "word": "172.25.164.254"}, 
            {"entity_group": "os_type", "start": 965, "end": 979, "word": "Windows Device"}, 
            {"entity_group": "ip_dest", "start": 372, "end": 385, "word": "208.91.112.53"}, 
            {"entity_group": "ip_dest", "start": 699, "end": 711, "word": "84.19.71.108"}, 
            {"entity_group": "ip_src", "start": 267, "end": 279, "word": "172.30.104.1"}, 
            {"entity_group": "timestamp", "start": 252, "end": 262, "word": "1650553379"}, 
            {"entity_group": "interface_dest", "start": 417, "end": 427, "word": "root-vlab1"}, 
            {"entity_group": "event_source", "start": 935, "end": 945, "word": "Windows PC"}, 
            {"entity_group": "event_source", "start": 51, "end": 60, "word": "Fortigate"}, 
            {"entity_group": "interface_src", "start": 335, "end": 344, "word": "vmservers"}, 
            {"entity_group": "event_source", "start": 42, "end": 50, "word": "Fortinet"}, 
            {"entity_group": "geoip_src", "start": 645, "end": 653, "word": "Reserved"}, 
            {"entity_group": "severity", "start": 818, "end": 826, "word": "elevated"}, 
            {"entity_group": "action", "start": 192, "end": 199, "word": "forward"}, 
            {"entity_group": "os_version", "start": 61, "end": 67, "word": "v6.0.9"}, 
            {"entity_group": "action", "start": 544, "end": 550, "word": "accept"}, 
            {"entity_group": "event_level", "start": 213, "end": 219, "word": "notice"}, 
            {"entity_group": "mask", "start": 620, "end": 626, "word": "<mask>"}, 
            {"entity_group": "event_id", "start": 68, "end": 73, "word": "00013"}, 
            {"entity_group": "process_id", "start": 752, "end": 757, "word": "16195"}, 
            {"entity_group": "tags", "start": 364, "end": 367, "word": "lan"}, 
            {"entity_group": "port_dest", "start": 390, "end": 392, "word": "53"}, 
            {"entity_group": "severity", "start": 97, "end": 98, "word": "3"}
            ], 
            "payload": "<117>Apr 21 17:02:59 172.25.164.254 CEF:0|Fortinet|Fortigate|v6.0.9|00013|traffic:forward accept|3|deviceExternalId=FG5H0E5819908089 FTNTFGTlogid=0000000013 cat=traffic:forward FTNTFGTsubtype=forward FTNTFGTlevel=notice FTNTFGTvd=vlab FTNTFGTeventtime=1650553379 src=172.30.104.1 shost=WIN-7OCV2UEAUKA spt=55250 deviceInboundInterface=vmservers FTNTFGTsrcintfrole=lan dst=208.91.112.53 dpt=53 deviceOutboundInterface=root-vlab1 FTNTFGTdstintfrole=undefined FTNTFGTpoluuid=d0d4bdd2-4d95-51ea-f827-483b4dae641a externalId=3341608789 proto=17 act=accept FTNTFGTpolicyid=7 FTNTFGTpolicytype=policy app=DNS FTNTFGTdstcountry=<mask> FTNTFGTsrccountry=Reserved FTNTFGTtrandisp=snat sourceTranslatedAddress=84.19.71.108 sourceTranslatedPort=55250 FTNTFGTappid=16195 FTNTFGTapp=DNS FTNTFGTappcat=Network.Service FTNTFGTapprisk=elevated FTNTFGTapplist=g-default FTNTFGTduration=180 out=84 in=148 FTNTFGTsentpkt=1 FTNTFGTrcvdpkt=1 FTNTFGTdevtype=Windows PC FTNTFGTdevcategory=Windows Device FTNTFGTosname=Windows 10 / 2016 FTNTFGTmastersrcmac=00:0c:29:87:df:ae FTNTFGTsrcmac=00:0c:29:87:df:ae FTNTFGTsrcserver=1"
            },
    ]   

    keep_mask = True if args.keep_mask else False 
    print_entities = args.print_entities if args.print_entities else False

    # Initialize the augmentator based on the selected mode
    if args.mode == "api":                  # API-based augmentation
        if keep_mask:
            host = args.API_host            # Ollama server host
            model = args.API_model          # Ollama model name
        else:
            host = None
            model = None

        # Initialize the DeepOllamaAugmentator
        generator = DeepOllamaAugmentator(host, model, keep_mask=keep_mask)
        try:
            augmented_data = generator.deep_augmentate(data_input, keep_mask=keep_mask, print_entities=print_entities)
        except Exception as e:
            logger.error(f"Error during augmentation: {e}")
            print(f"Error during augmentation: {e}")
            exit(1)

    elif args.mode == "deep":                                           # Deep augmentation using MLflow model
        if keep_mask:
            deep_model_name = args.deep_model_name                      # MLflow model name
            deep_model_alias = args.deep_model_alias                    # MLflow model alias

            mlflow.set_tracking_uri(uri="http://192.168.40.5:5000/")    # Update URI if needed
            client = MlflowClient()
            import sectech_log_augmentator.log_config as log_config

            model_version_info = client.get_model_version_by_alias(deep_model_name, deep_model_alias)
            model_version = model_version_info.version

            model_uri = f"models:/{deep_model_name}/{model_version}"

            local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
            print(f"Model downloaded to: {local_model_path}")

            model_path = f"{local_model_path}/model"
            tokenizer_path = f"{local_model_path}/components/tokenizer"

        else:
            model_path = None
            tokenizer_path = None

        # Initialize the DeepAugmentator
        deep = DeepAugmentator(model_path=model_path, tokenizer_path=tokenizer_path, keep_mask=keep_mask)
        try:
            augmented_data = deep.deep_augmentate(data_input, keep_mask=keep_mask, print_entities=print_entities)
        except Exception as e:
            logger.error(f"Error during augmentation: {e}")
            print(f"Error during augmentation: {e}")
            exit(1)
