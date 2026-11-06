# Biofouling Assistant LoRA Adapter Directory

This directory is designated for PEFT LoRA adapter weights fine-tuned on Industrial Biofouling Detection and Marine Inspection dialogues.

When adapter weights (`adapter_config.json` and `adapter_model.safetensors` / `adapter_model.bin`) are placed in this folder, `BioAssistChatbot` will automatically detect and attach them to the base `Qwen/Qwen2.5-1.5B-Instruct` model.

If no adapter weights are present, `BioAssistChatbot` operates directly on the base model conditioned by the domain-specific system prompt and offline inspection knowledge base.
