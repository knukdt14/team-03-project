import os
import sys
import torch
import ssl
from typing import Optional, Any

# Fix Windows SSL Certificate Store error (_ssl.c:4057)
_orig_load_default_certs = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except Exception:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.config import LOCAL_LLM_MODEL, LLM_MODE

# Import HuggingFace integrations
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LLMFactory:
    """
    Modular LLM Factory:
    - Mode 'local': Downloads model via HuggingFace transformers and runs locally on PyTorch (CUDA / CPU).
    - Mode 'api': Calls HuggingFace Serverless Inference API.
    - Easy one-switch configuration between Local Execution and API Execution.
    """

    @staticmethod
    def get_llm(model_name: str = LOCAL_LLM_MODEL, mode: str = LLM_MODE) -> Any:
        print(f"[LLMFactory] Initializing LLM '{model_name}' in mode='{mode}'...")
        
        if mode == "local" and HAS_TRANSFORMERS:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[LLMFactory Local] Loading model '{model_name}' on device '{device}'...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                # Torch dtype optimization
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    # A compact answer budget reduces Qwen-0.5B's tendency to
                    # continue with unsupported procedural explanations.
                    max_new_tokens=96,
                    temperature=0.0,
                    do_sample=False
                )
                
                hf_pipe = HuggingFacePipeline(pipeline=pipe)
                return ChatHuggingFace(llm=hf_pipe)
            except Exception as e:
                print(f"[LLMFactory Warning] Local model loading failed: {e}. Falling back to API mode.")
                
        # API Mode or Fallback
        if HAS_TRANSFORMERS:
            try:
                hf_token = os.getenv("HF_TOKEN")
                endpoint = HuggingFaceEndpoint(
                    repo_id=model_name,
                    task="text-generation",
                    max_new_tokens=96,
                    temperature=0.0,
                    huggingfacehub_api_token=hf_token
                )
                return ChatHuggingFace(llm=endpoint)
            except Exception as e:
                print(f"[LLMFactory Warning] API mode failed: {e}.")
                
        # Fallback
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


if __name__ == "__main__":
    llm = LLMFactory.get_llm(model_name="Qwen/Qwen2.5-3B-Instruct", mode="api")
    print(f"LLM initialized: {type(llm)}")
