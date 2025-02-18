import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import folder_paths

# 添加any_type导入（ComfyUI内置类型）
any_type = type("any_type", (object,), {})()

def tensor_to_pil(image_tensor, batch_index=0) -> Image.Image:
    single_image = image_tensor[batch_index].numpy()
    if single_image.max() <= 1.0:
        single_image = (single_image * 255).astype(np.uint8)
    else:
        single_image = single_image.astype(np.uint8)
    return Image.fromarray(single_image.squeeze())

class Qwen2VL:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cuda:1", "cuda:0"], {"default": "cuda"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"], {"default": "Qwen2.5-VL-3B-Instruct"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 1}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("final_result", "list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(self, device, text, model, quantization, keep_model_loaded, temperature, max_new_tokens, seed, image=None):
        results = []
        try:
            if seed != -1:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            selected_device = torch.device(device)
            model_id = f"qwen/{model}"
            self.model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", os.path.basename(model_id))

            if not os.path.exists(self.model_checkpoint):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint, local_dir_use_symlinks=False)

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_checkpoint, 
                    min_pixels=256*28*28,
                    max_pixels=1024*28*28
                )

            if self.model is None:
                quantization_config = None
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map={"": device},
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True
                )

            if image is not None and torch.is_tensor(image):
                batch_size = image.shape[0]
                for idx in range(batch_size):
                    pil_img = tensor_to_pil(image, idx)
                    message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": text}
                        ]
                    }
                    single_text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info([message])
                    inputs = self.processor(
                        text=[single_text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt"
                    ).to(selected_device)
                    
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                    result = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    results.append(result)
                    del inputs
                    torch.cuda.empty_cache()
            else:
                message = {"role": "user", "content": [{"type": "text", "text": text}]}
                single_text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[single_text], return_tensors="pt").to(selected_device)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                results.append(result)

            final_result = "\n----------------\n".join([f"结果 {i+1}:\n{res}" for i, res in enumerate(results)])

        except Exception as e:
            return (f"推理出错: {str(e)}", [])

        if not keep_model_loaded:
            del self.processor
            del self.model
            self.processor = None
            self.model = None
            torch.cuda.empty_cache()

        return (final_result, results)

class Qwen2:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cuda:1", "cuda:0"], {"default": "cuda:1"}),
                "system": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"], {"default": "Qwen2.5-7B-Instruct"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 1}),
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("final_result", "list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(self, device, system, prompt, model, quantization, keep_model_loaded, temperature, max_new_tokens, seed):
        results = []
        try:
            if not prompt.strip():
                return ("错误：输入内容为空", [])

            if seed != -1:
                torch.manual_seed(seed)

            selected_device = torch.device(device)
            model_id = f"qwen/{model}"
            self.model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", os.path.basename(model_id))

            if not os.path.exists(self.model_checkpoint):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_id, local_dir=self.model_checkpoint, local_dir_use_symlinks=False)

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

            if self.model is None:
                quantization_config = None
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_checkpoint,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map={"": device},
                    quantization_config=quantization_config,
                )

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(selected_device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            results = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            final_result = "\n----------------\n".join([f"结果 {i+1}:\n{res}" for i, res in enumerate(results)])

        except Exception as e:
            return (f"推理出错: {str(e)}", [])

        if not keep_model_loaded:
            del self.tokenizer
            del self.model
            self.tokenizer = None
            self.model = None
            torch.cuda.empty_cache()

        return (final_result, results)
