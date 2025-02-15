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


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # 将形状为 [batch, height, width, channels] 的 tensor 中指定批次转换为 PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


class Qwen2VL:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cuda:1", "cuda:0"], {"default": "cuda:1",},),
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-VL-3B-Instruct",
                        "Qwen2.5-VL-7B-Instruct",
                    ],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 1}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"
    
    def inference(
        self,
        device,
        text,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        image=None,
    ):
        # 设置随机种子（如果指定）
        if seed != -1:
            torch.manual_seed(seed)
        
        # 将用户选择的设备转换为 torch.device 对象
        selected_device = torch.device(device)
    
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )
    
        # 模型不存在则自动下载
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )
    
        # 初始化图像处理器
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28
            )
    
        # 初始化模型
        if self.model is None:
            quantization_config = None
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
            # 通过 device_map 参数将模型加载到指定设备
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map={"": device},
                quantization_config=quantization_config,
            )
    
        # 处理输入数据
        messages_list = []
        if torch.is_tensor(image):
            batch_size = image.shape[0]
            pil_images = [tensor_to_pil(image, i) for i in range(batch_size)]
            for img in pil_images:
                messages_list.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text}
                    ]
                })
        else:
            messages_list.append({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })
    
        try:
            texts = [
                self.processor.apply_chat_template(
                    [msg],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for msg in messages_list
            ]
            image_inputs, video_inputs = process_vision_info(messages_list)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(selected_device)
    
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
    
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
    
            results = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
    
            final_result = "\n----------------\n".join([
                f"Result {i+1}:\n{res}" for i, res in enumerate(results)
            ])
    
        except Exception as e:
            return (f"Error during inference: {str(e)}",)
    
        # 清理资源（如果不需要保持模型加载）
        if not keep_model_loaded:
            del self.processor
            del self.model
            self.processor = None
            self.model = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
        return (final_result,)


class Qwen2:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cuda:1", "cuda:0"], {"default": "cuda:1",},),
                "system": (
                    "STRING",
                    {"default": "You are a helpful assistant.", "multiline": True},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                    ],
                    {"default": "Qwen2.5-7B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        device,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
    ):
        if not prompt.strip():
            return ("Error: Prompt input is empty.",)

        if seed != -1:
            torch.manual_seed(seed)

        selected_device = torch.device(device)
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        if self.model is None:
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map={"": device},
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(selected_device)
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.tokenizer
                del self.model
                self.tokenizer = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return result
