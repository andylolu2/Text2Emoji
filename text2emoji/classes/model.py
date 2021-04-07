from typing import List, Tuple
import logging

from transformers import DistilBertConfig, DistilBertTokenizerFast
import torch
import numpy as np

from text2emoji.helpers import create_model_for_provider
from config import TrainConfig


class Text2EmojiModel(torch.nn.Module):
    EOS_TOKEN = TrainConfig.EOS_TOKEN
    sentence_onnx = create_model_for_provider(
        TrainConfig.SENTENCE_ONNX_PATH, "CUDAExecutionProvider")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        TrainConfig.DISTILBERT_NAME)

    def __init__(self, emoji_vocab: list, device: torch.device):
        super(Text2EmojiModel, self).__init__()
        self.emoji_vocab = emoji_vocab + [self.EOS_TOKEN]
        self.config = DistilBertConfig()
        self.encoder_head = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.config.dim,
                nhead=self.config.n_heads,
            ),
            num_layers=1,
        )
        self.classification = torch.nn.Linear(
            self.config.dim, len(self.emoji_vocab))
        self.device = device

    def forward(self, x, mask):
        '''
        args:
            x: distilbert embeddings of shape (SEQ_LEN, batch_size, HIDDEN_DIM)
            mask: attention mask of shape (batch_size, SEQ_LEN)

        returns:
            token classification logits: (max_seq_len, batch_size, emoji_vocab_size)
        '''
        mask = ~mask.bool()
        # Note: Huggingface masks attends to 1 and ignore 0, pytorch attends to True and ignores False
        _x = self.encoder_head(
            src=x,
            src_key_padding_mask=mask,
        )
        # print(_x.shape)  # (SEQ_LEN, batch_size, HIDDEN_DIM)
        zeros = torch.zeros(
            min(TrainConfig.MAX_EMOJI_SENTENCE_LEN -
                _x.shape[0], _x.shape[0]),
            _x.shape[1],
            _x.shape[2],
            device=self.device,
            requires_grad=True
        )
        _x = torch.cat(
            (_x, zeros),
            dim=0,
        )
        # print(_x.shape)  # (max_seq_len, batch_size, HIDDEN_DIM)
        _x = self.classification(_x)
        # print(_x.shape)  # (max_seq_len, batch_size, emoji_vocab_size)
        return _x

    def embedding_to_emojis(
            self,
            x: torch.Tensor,
            smaples: int = 1,
            debug: bool = False) -> Tuple[List[str], torch.Tensor]:
        '''
        args:
            x: one of self.forward outputs of shape (max_seq_len, emoji_vocab_size)

        returns:
            list[str], float of shape (<=max_seq_len,), 0
        '''
        if debug:
            torch.manual_seed(0)
        output = []
        m = torch.distributions.categorical.Categorical(
            logits=x, validate_args=True)
        s = m.sample()
        log_prob = m.log_prob(s)
        last = len(s)
        logging.debug("embedding_to_emojis: ")
        logging.debug(f"sample: {s}")
        for i in range(len(s)):
            token_id = s[i]
            token = self.emoji_vocab[token_id]
            logging.debug(f"token: {token}, token_id: {token_id}")
            if token == self.EOS_TOKEN:
                last = i + 1
                break
            output.append(token)
        logging.debug(f"last: {last}")
        return (output, torch.unsqueeze(torch.sum(log_prob[:last]), -1))

    def seq_to_emojis(self, seq):
        output = []
        last = len(seq)
        for i, tok_id in enumerate(seq):
            if tok_id == len(self.emoji_vocab) - 1:
                last = i
                break
            output.append(self.emoji_vocab[tok_id])
        return output, last

    def embeddings_to_emojis(
            self,
            x: torch.Tensor,
            sample_size: int = 1,
            debug: bool = False) -> Tuple[List[List[List[str]]], torch.Tensor]:
        '''
        args:
            x: self.forward outputs of shape (max_seq_len, batch_size, emoji_vocab_size)

        returns:
            List[List[List[str]]], 2d tensor of shape (batch_size, samples, <=max_seq_len), (batch_size, samples)
        '''
        logging.debug("embeddings_to_emojis: ")
        logging.debug(f"x shape: {x.shape}")
        # Need (batch_size, max_seq_len, emoji_vocab_size) for Categorical
        x = torch.transpose(x, 0, 1)
        dist = torch.distributions.Categorical(logits=x)

        sample = dist.sample((sample_size,))
        # sample shape: (sample_size, batch_size, max_seq_len)
        log_probs = dist.log_prob(sample)
        # log_probs shape: (sample_size, batch_size, max_seq_len)

        sample = torch.transpose(sample, 0, 1)
        # sample shape: (batch_size, sample_size, max_seq_len)
        log_probs = torch.transpose(log_probs, 0, 1)
        # log_probs shape: (batch_size, sample_size, max_seq_len)

        emojis = []
        eos_pos = []
        for batch in sample:
            batch_emoji = []
            batch_last = []
            for s in batch:
                emoji, last = self.seq_to_emojis(s)
                batch_emoji.append(emoji)
                batch_last.append(last)
            emojis.append(batch_emoji)
            eos_pos.append(batch_last)
        eos_pos = torch.tensor(eos_pos).to(self.device)

        mask = torch.arange(sample.shape[-1]).to(self.device).view(1, sample.shape[-1])\
            .repeat(sample.shape[0], sample.shape[1], 1) < eos_pos.view(sample.shape[0], sample.shape[1], 1)
        log_probs_sum = (log_probs * mask).sum(-1)
        logging.debug(f"log_probs_sum shape: {log_probs_sum.shape}")
        return emojis, log_probs_sum

    @ classmethod
    def emojis_to_sentence_embedding(self, x: List[List[List[str]]], debug: bool = False) -> np.ndarray:
        '''
        args:
            x: embeddings_to_emojis outputs of shape (batch_size, sample_size, max_seq_len)

        returns:
            array of shape (batch_size, sample_size, sentence_embedding_dim)
        '''
        logging.debug("emojis_to_sentence_embeddings: ")

        batch_size = len(x)
        sample_size = len(x[0])
        logging.debug(f"Batch_size: {batch_size}")
        logging.debug(f"Sample_size: {sample_size}")

        flattened_x = [" ".join(seq) for batch in x for seq in batch]
        # flatten for parallel computation
        # flattened_x shape: (batch_size * sample_size)
        inputs = self.tokenizer(
            flattened_x,
            padding=True,
            truncation=True,
            return_tensors='np'
        )
        _x = self.sentence_onnx.run(None, dict(inputs))
        _x = self.mean_pooling(_x, inputs['attention_mask'])
        # _x shape: (batch_size * sample_size, sentence_embedding_dim)
        _x = _x.reshape(batch_size, sample_size, _x.shape[-1])
        return _x

    def embeddings_to_emojis_to_sentence_embeddings(
            self,
            x: torch.Tensor,
            sample_size: int = 1,
            debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:

        emojis, log_prob = self.embeddings_to_emojis(
            x, sample_size=sample_size, debug=debug)
        _x = self.emojis_to_sentence_embedding(emojis, debug=debug)
        return torch.from_numpy(_x).float().to(self.device), log_prob, emojis

    @ staticmethod
    def mean_pooling(x, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        args:
            x: sentence transformer last layer outputs of shape (1, batch_size, SEQ_LEN, HIDDEN_DIM_SIZE)

        returns:
            mean of last layer outpus across SEQ_LEN, taking attention mask into account
        '''
        token_embeddings = x[0]
        assert isinstance(token_embeddings, np.ndarray)
        if isinstance(attention_mask, np.ndarray):
            input_mask_expanded = np.tile(np.expand_dims(
                attention_mask, axis=-1), (1, 1, token_embeddings.shape[-1])).astype('float32')
        else:
            assert isinstance(attention_mask, torch.tensor)
            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(token_embeddings.shape).float()
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1),
                           a_max=None, a_min=1e-9)
        return sum_embeddings / sum_mask


if __name__ == "__main__":
    logging.root.setLevel(logging.DEBUG)
    dummy_vocab = ['a', 'b', 'c']
    dummy_input = torch.rand(64, 3, 768)
    dummy_mask = torch.cat((torch.ones(3, 32), torch.zeros(3, 32)), dim=1)
    model = Text2EmojiModel(dummy_vocab, 'cpu')
    output = model(dummy_input, dummy_mask)
    print(output.shape)
    embed, log_prob, emojis = model.embeddings_to_emojis_to_sentence_embeddings(
        output, sample_size=10)
    print(embed.shape)
    print(log_prob.shape)
    print(emojis)
