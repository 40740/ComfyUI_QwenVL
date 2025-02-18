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


def tensor_to_pil(image_tensor, batch_index=0) -> Image.Image:
    """
    将形状为 [batch, height, width, channels] 的 tensor 转换为 PIL Image
    参数:
        image_tensor: 输入的图像张量
        batch_index: 要转换的批次索引
    返回:
        PIL Image 对象
    """
    # 提取指定批次的图像并移除批次维度
    single_image = image_tensor[batch_index].numpy()
    
    # 将值域从 [0,1] 转换到 [0,255]
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
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cuda:1", "cuda:0"], {"default": "cuda"}),
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
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 设备选择
        selected_device = torch.device(device)
        
        # 模型路径处理
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        # 自动下载模型
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=self.model_checkpoint,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                return (f"模型下载失败: {str(e)}",)

        # 初始化处理器
        if self.processor is None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_checkpoint,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1024 * 28 * 28
                )
            except Exception as e:
                return (f"处理器初始化失败: {str(e)}",)

        # 初始化模型
        if self.model is None:
            quantization_config = None
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map={"": device},
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True  # 减少内存占用
                )
            except Exception as e:
                return (f"模型加载失败: {str(e)}",)

        # 结果收集列表
        results = []

        try:
            # 处理图像输入
            if image is not None and torch.is_tensor(image):
                batch_size = image.shape[0]
                
                # 逐张处理图片
                for idx in range(batch_size):
                    # 转换张量为PIL图像
                    pil_img = tensor_to_pil(image, idx)
                    
                    # 构建消息
                    message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": text}
                        ]
                    }
                    
                    # 处理单条消息
                    single_text = self.processor.apply_chat_template(
                        [message],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # 处理视觉信息
                    image_inputs, video_inputs = process_vision_info([message])
                    
                    # 准备输入（保持列表结构）
                    inputs = self.processor(
                        text=[single_text],  # 保持列表形式
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt"
                    ).to(selected_device)
                    
                    # 单样本推理
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    
                    # 解码结果
                    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                    result = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]  # 直接取第一个结果
                    
                    results.append(result)
                    
                    # 立即释放临时变量
                    del inputs
                    torch.cuda.empty_cache()
                    
            # 处理纯文本输入
            else:
                message = {
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                }
                
                single_text = self.processor.apply_chat_template(
                    [message],
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[single_text],
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

            # 格式化最终输出
            final_result = "\n----------------\n".join([
                f"结果 {i+1}:\n{res}" for i, res in enumerate(results)
            ])

        except Exception as e:
            return (f"推理过程中出错: {str(e)}",)

        # 清理资源
        if not keep_model_loaded:
            try:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
            except:
                pass

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
