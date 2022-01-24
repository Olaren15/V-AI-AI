from datetime import datetime

from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-large"
# model_name = "microsoft/DialoGPT-medium"
# model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


class Conversation:
    def __init__(self):
        self.step = 0
        self.last_interaction = datetime.now()

    def reply(self, message):
        self.last_interaction = datetime.now()

        # encode the input and add end of string token
        self.input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
        # concatenate new user input with chat history (if there is)
        self.bot_input_ids = torch.cat([self.chat_history_ids, self.input_ids],
                                       dim=-1) if self.step > 0 else self.input_ids

        if self.step > 4:
            splits = torch.split(self.bot_input_ids, split_size_or_sections=40, dim=-1)

            print(f"splits: {splits}")

            self.bot_input_ids = torch.cat([splits[-2], splits[-1]], dim=-1)

            print(f"choosen: {self.bot_input_ids}")

        # generate a bot response
        self.chat_history_ids = model.generate(
            self.bot_input_ids,
            max_length=1000,
            do_sample=True,
            top_p=0.95,
            top_k=200,
            temperature=1.0,
            length_penalty=1,
            pad_token_id=tokenizer.eos_token_id
        )

        print(self.chat_history_ids)

        self.step += 1
        # return the outputs
        reply = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        return reply
