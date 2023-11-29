import os
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from src.utils.file_io import hack_path
import pytorch_lightning as pl

class T5TextEncoder(pl.LightningModule):

    def __init__(
            self,
            modelpath: str,
            finetune: bool = False,
            **kwargs
        ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        from transformers import logging
        from transformers import AutoModel, AutoTokenizer
        from transformers import AutoTokenizer, T5TextEncoderModel

        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(hack_path(modelpath), 
                                                       legacy=True)
        self.language_model = T5TextEncoderModel.from_pretrained(
                hack_path(modelpath))
        self.language_model.resize_token_embeddings(len(self.tokenizer))

        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

    def forward(
        self,
        texts: List[str],
    ):
        inputs = texts
        outputs = texts

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")


        input_ids = tokenizer(texts, return_tensors="pt").input_ids  # Batch size 1
        text_input_ids = text_inputs.input_ids.to(self.text_model.device)
        txt_att_mask = text_inputs.attention_mask.to(self.text_model.device)

        outputs = self.language_model(input_ids=text_input_ids, attention_mask=txt_att_mask)
        last_hidden_states = outputs.last_hidden_state

        return outputs