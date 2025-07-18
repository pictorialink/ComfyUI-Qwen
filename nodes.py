from pathlib import Path
import os
import json
import folder_paths
import torch
import re
from PIL import Image
import numpy as np
import gc

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    raise RuntimeError("No suitable device found. Please ensure MPS or CUDA is available.")

def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

class ModelsInfo:
    def __init__(self):
        current_dir = Path(__file__).parent.resolve()
        models_info_file = os.path.join(current_dir, "models.json")
        with open(models_info_file, "r", encoding="utf-8") as f:
            self.models_info = json.load(f)

class Qwen25VL(ModelsInfo):
    def __init__(self):
        super().__init__()
        self.model_checkpoint = None
        self.model = None
        self.processor = None
        if device == "cuda":
            self.bf16_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(device)[0] >= 8
            )
        else:
            self.config = None
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {
                    "default": 0,  # 默认值
                    "min": 0,      # 最小值
                    "max": 0xffffffffffffffff,  # 最大值（64位整数）
                    "step": 1      # 步长
                }),
            },
            "optional": {
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_mps" if device == "mps" else "inference_cuda"
    CATEGORY = "Comfyui_Qwen"

    def inference_mps(
        self,
        image,
        text: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        seed: int = -1,
        quantization: str = "none",
    ):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        import mlx.core as mx
        from mlx_vlm import load,generate
        model_info = self.models_info["mps"][1]
        self.model_checkpoint = os.path.join(folder_paths.base_path, model_info['local_path'])
        # 如果模型不存在就下载
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            # 使用 huggingface 下载
            file_path = snapshot_download(repo_id=model_info['repo_id'], local_dir=self.model_checkpoint,local_dir_use_symlinks=False)
            print(f"Model downloaded to: {file_path}")
        # 加载模型
        mx.random.seed(seed=seed)
        self.model, self.processor = load(self.model_checkpoint)
        self.config = load_config(self.model_checkpoint)
        formatted_prompt = apply_chat_template(
            self.processor, self.config, text, num_images=1
        )
        # 推理
        image = tensor_to_pil(image)
        print(f"image type: {type(image)}")
        output = generate(self.model, self.processor, formatted_prompt, [image], verbose=False, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
        del self.model, self.processor, self.config  # 清理显存
        self.model = None
        self.processor = None
        self.config = None
        torch.mps.empty_cache()  # 清理 MPS 显存
        print(f"Output: {output}")
        return (output.text,)
    
    def inference_cuda(
        self,
        text,
        quantization,
        temperature,
        max_new_tokens,
        seed,
        image=None,
    ):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from qwen_vl_utils import process_vision_info
        if seed != -1:
            torch.manual_seed(seed)
        # 模型是否存在
        model = self.models_info['cuda'][2]
        self.model_checkpoint = os.path.join(folder_paths.base_path, model['local_path'])
        if not os.path.exists(self.model_checkpoint):
                from huggingface_hub import snapshot_download
                # 使用 huggingface 下载
                file_path = snapshot_download(repo_id=model['repo_id'], local_dir=self.model_checkpoint,local_dir_use_symlinks=False)
                print(f"Model downloaded to: {file_path}")
            
            
        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
            print("deal image")
            pil_image = tensor_to_pil(image)
            messages[0]["content"].insert(0, {
                "type": "image",
                "image": pil_image,
            })

            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("deal messages", messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # 推理
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)

            del self.processor
            del self.model
            self.processor = None
            self.model = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            return (result[0],) if result else ("",)
        
    
class Qwen3(ModelsInfo):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        if device == "cuda":
            gguf_info = self.models_info['cuda'][0]
            tokenizer_info = self.models_info['cuda'][1]
            self.gguf_file = os.path.join(folder_paths.base_path, gguf_info['local_path'], gguf_info['files'][0])
            self.tokenizer_dir = os.path.join(folder_paths.base_path, tokenizer_info['local_path'])
        else:
            self.model_checkpoint = None
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "你是一个智能助理","multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "direct": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {
                    "default": 0,  # 默认值
                    "min": 0,      # 最小值
                    "max": 0xffffffffffffffff,  # 最大值（64位整数）
                    "step": 1      # 步长
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_mps" if device == "mps" else "inference_cuda"
    CATEGORY = "Comfyui_Qwen"

    def inference_mps(
        self,
        system_prompt: str,
        user_prompt: str,
        direct: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        seed: int = -1,
    ):
        if direct:
            return (user_prompt,)
        model_info = self.models_info["mps"][0]
        self.model_checkpoint = os.path.join(folder_paths.base_path, model_info['local_path'])
        # 如果模型不存在就下载
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            # 使用 huggingface 下载
            file_path = snapshot_download(repo_id=model_info['repo_id'], local_dir=self.model_chnaeckpoint,local_dir_use_symlinks=False)
            print(f"Model downloaded to: {file_path}")
        # 加载模型
        from mlx_lm import load, generate
        import mlx.core as mx
        mx.random.seed(seed=seed)
        self.model, self.tokenizer = load(self.model_checkpoint)
        prompt = f"{system_prompt}\n\n{user_prompt} /no_think"  # 添加 /no_think 标签
        if self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt} /no_think"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

        # 5. 生成响应
        def remove_think_tags(text):
            """
            删除文本中所有的<think>标签及其内容，并移除标签后的换行符
            
            参数:
            text (str): 包含XML标签的文本
            
            返回:
            str: 移除<think>标签及后续换行符后的文本
            """
            # 正则表达式模式：匹配<think>标签及其内容，以及紧随其后的换行符
            pattern = r'<think>.*?</think>\s*'
            # 使用re.DOTALL标志使.可以匹配换行符
            # 使用非贪婪匹配(.*?)确保只匹配到最近的</think>
            return re.sub(pattern, '', text, flags=re.DOTALL)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=True,
        )
        del self.model, self.tokenizer  # 清理显存
        self.model = None
        self.tokenizer = None
        torch.mps.empty_cache()  # 清理 MPS 显存
        print(f"Response: {response}")
        return (remove_think_tags(response),)
    
    def inference_cuda(
        self,
        system_prompt,
        user_prompt,
        direct,
        # keep_model_loaded,
        temperature,
        max_new_tokens,
        seed=-1
    ):
        from huggingface_hub import hf_hub_download
        import pynvml
        def get_free_vram():
            if torch.cuda.is_available():
                pynvml.nvmlInit()
                # 这里的0是GPU id
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"总显存: {meminfo.total/1024**3:.2f} GB, 已用显存: {meminfo.used/1024**3:.2f} GB, 剩余显存: {meminfo.free/1024**3:.2f} GB")
                return meminfo.free
            else:
                print("未检测到 GPU")
                return 0
        if direct:
            return (user_prompt,)
        # 模型是否存在
        models = self.models_info['cuda'][:2]
        for model in models:
            file_dir = os.path.join(folder_paths.base_path, model['local_path'])
            for file_name in model['files']:
                if not os.path.exists(os.path.join(file_dir, file_name)):
                    # 使用 huggingface 下载
                    file_path = hf_hub_download(
                        repo_id=model['repo_id'],
                        filename=file_name,
                        local_dir=file_dir
                    )
                    print(f"Model downloaded to: {file_path}")
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        # 3. 格式化提示（Qwen3 使用特定的聊天模板）
        def format_prompt(system_prompt,user_prompt):
            # 禁用 think 模式
            user_prompt += " /no_think"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        # 检查当前显存是否能够加载
        if get_free_vram() < 8 * 1024**3:
            import comfy.model_management as mm
            gc.collect()
            mm.unload_all_models()
            mm.soft_empty_cache()
        if self.model is None:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.gguf_file,
                n_ctx=4096,  # 上下文长度
                n_gpu_layers=-1,  # -1 表示将所有层卸载到 GPU（如果支持）
                temperature=temperature,  # 控制生成文本的随机性
                max_tokens=max_new_tokens,  # 最大生成 token 数
                verbose=False,
            )
        prompt = format_prompt(system_prompt, user_prompt)
        # 5. 生成响应
        def remove_think_tags(text):
            """
            删除文本中所有的<think>标签及其内容，并移除标签后的换行符
            
            参数:
            text (str): 包含XML标签的文本
            
            返回:
            str: 移除<think>标签及后续换行符后的文本
            """
            # 正则表达式模式：匹配<think>标签及其内容，以及紧随其后的换行符
            pattern = r'<think>.*?</think>\s*'
            # 使用re.DOTALL标志使.可以匹配换行符
            # 使用非贪婪匹配(.*?)确保只匹配到最近的</think>
            return re.sub(pattern, '', text, flags=re.DOTALL)
        
        output = self.model(
            prompt,
            max_tokens=2048,
            stop=["</s>"],
            echo=False,
            seed=seed
        )
        response = remove_think_tags(output["choices"][0]["text"])
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return (response,)