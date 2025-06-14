from dataclasses import dataclass, field
import os
# upgrade flash attention here
try:
    os.system("pip install flash-attn --no-build-isolation --upgrade")
except:
    print("flash-attn failed to install")

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from peft.tuners.lora import LoraLayer
from datasets import load_dataset
from huggingface_hub import login
import torch
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback,
)
import bitsandbytes


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bitsandbytes.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    modules = find_all_linear_names(model)
    print(f'Number of modules to quantize : {len(modules)}')

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model


def tokenize_fn(example):
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token="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",
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=2048)
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def training_loop(script_args, training_args):
    print("/opt/ml/input/data/training/train_dataset.jsonl Size (bytes):", os.path.getsize("/opt/ml/input/data/training/train_dataset.jsonl"))
    train_dataset = load_dataset("json", data_files="/opt/ml/input/data/training/train_dataset.jsonl")["train"]
    print("Training Dataset loaded from :", "train_dataset.jsonl")
    print("Training dataset count :", len(train_dataset))
    tokenized_training_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)

    print("/opt/ml/input/data/validation/validation_dataset.jsonl Size (bytes):", os.path.getsize("/opt/ml/input/data/validation/validation_dataset.jsonl"))
    validation_dataset = load_dataset("json", data_files="/opt/ml/input/data/validation/validation_dataset.jsonl")["train"]
    print("Validation Dataset loaded from :", "validation_dataset.jsonl")
    print("Validation dataset count :", len(validation_dataset))
    tokenized_validation_dataset = validation_dataset.map(tokenize_fn, batched=True, remove_columns=validation_dataset.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        device_map="auto",
        use_flash_attention_2=script_args.use_flash_attn,
        quantization_config=bnb_config,
    )

    model = create_peft_model(
        model, gradient_checkpointing=training_args.gradient_checkpointing, bf16=training_args.bf16
    )

    # Creating trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    sagemaker_save_dir = "/opt/ml/model/"
    if script_args.merge_adapters:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB")
    else:
        trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)

    # save the tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, padding_side="left")
    tokenizer.save_pretrained(sagemaker_save_dir)


@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default=None,
    )
    hf_token: Optional[str] = field(default=None, metadata={"help": "Hugging Face token for authentication"})
    trust_remote_code: bool = field(
        metadata={"help": "Whether to trust remote code."},
        default=False,
    )
    use_flash_attn: bool = field(
        metadata={"help": "Whether to use Flash Attention."},
        default=False,
    )
    merge_adapters: bool = field(
        metadata={"help": "Whether to merge weights for LoRA."},
        default=False,
    )


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(training_args.seed)

    # login to hub
    token = script_args.hf_token if script_args.hf_token else os.getenv("HF_TOKEN", None)
    if token:
        print(f"Logging into the Hugging Face Hub with token {token[:10]}...")
        login(token=token)

    # run training function
    training_loop(script_args, training_args)


if __name__ == "__main__":
    main()