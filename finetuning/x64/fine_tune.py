import torch
import transformers
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

import bitsandbytes as bnb
from trl import SFTTrainer

from finetuning.x64.env import hf_token


class GemmaPrompt:
  def __init__(self):
    self.user_instrs = []
    self.output_instrs = []
  def add_user_instr(self, user_instr='', inputs=''):
    self.user_instrs.append(f'<start_of_turn>user {user_instr}. Patient: {inputs}<end_of_turn>')
  def add_output_instr(self, output_instr):
    self.output_instrs.append(f'<start_of_turn>model {output_instr}<end_of_turn>')
  def __str__(self):
    return "".join(self.user_instrs) + '\n' + "".join(self.output_instrs)

class ModelTraining:
    HF_TOKEN = hf_token
    def __init__(self, model_name='gemma2:2b', dataset_name=''):
        self.model_name = model_name
        self.dataset_name = dataset_name
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto') if torch.cuda.is_available() else AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
        self.dataset = load_dataset(dataset_name)
        text_column = [self._generate_prompt(data_point) for data_point in self.dataset["train"]]
        new_dataset = self.dataset["train"].add_column("prompt", text_column)
        dataset = new_dataset.shuffle(seed=1234)  # Shuffle dataset here
        self.dataset = dataset.map(lambda samples: self.tokenizer(samples["prompt"]), batched=True)
        self.modules = self._find_all_linear_names(self.model)
        self.model.gradient_checkpointing_enable()
        # preprocess quantized model for training
        model = prepare_model_for_kbit_training(self.model)
        self.lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=self.modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, self.lora_config)
        self.trainer = None

    def train(self, training_args, test_ratio=0.1):
        dataset = self.dataset.train_test_split(test_size=test_ratio)
        train_data = dataset["train"]
        test_data = dataset["test"]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        training_args = training_args if training_args else transformers.TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4,
                                                       gradient_checkpointing=True, max_steps=100, learning_rate=2e-4,
                                                       logging_steps=100, output_dir="outputs",
                                                       optim="paged_adamw_32bit",
                                                       save_strategy="epoch", num_train_epochs=1,
                                                       lr_scheduler_type="cosine", )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=test_data,
            dataset_text_field="prompt",
            peft_config=self.lora_config,
            max_seq_length=2500,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        torch.cuda.empty_cache()
        trainer.train()
        self.trainer = trainer
        return trainer

    def save_and_publish(self, new_model_name="gemma-2-2b-chatdoctor"):
        # Your existing code for loading and merging the model
        # save model
        self.trainer.model.save_pretrained(new_model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        merged_model = PeftModel.from_pretrained(base_model, new_model_name)
        merged_model = merged_model.merge_and_unload()

        # Save the merged model
        # save_adapter=True, save_config=True
        merged_model.save_pretrained("merged_model", safe_serialization=True)
        self.tokenizer.save_pretrained("merged_model")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        merged_model.push_to_hub(new_model_name, use_temp_dir=False, private=True)
        self.tokenizer.push_to_hub(new_model_name, use_temp_dir=False, private=True)
        return True

    def _generate_prompt(self, data_point):
        prompt = GemmaPrompt()
        prompt.add_user_instr(user_instr=data_point['instruction'], inputs=data_point['input'])
        prompt.add_output_instr(data_point['output'])
        return str(prompt)

    def _find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            # needed for 16-bit
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
        return list(lora_module_names)
    #
