import torch
import transformers

from config import DreamerConfig


class Text2EmojiDreamer(torch.nn.Module):
    def __init__(self, device):
        super(Text2EmojiDreamer, self).__init__()
        self.device = device
        self.sentence_transformer = transformers.AutoModel.from_pretrained(
            DreamerConfig.SENTENCE_TRANSFORMER_NAME)
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, mask, is_onehot=True):
        '''calculate sentence embeddings from inputs_ids

        Args:
            input_ids: either onehot-encoded or normal input_ids,
                flagged by 'is_onehot'
            mask: attention mask
            is_onehot: whethere 'input_ids' is onehot or not. Default True.

        Returns:
            _x: sentence embeddings
        '''
        assert isinstance(input_ids, torch.Tensor)
        if not is_onehot:
            assert len(input_ids.shape) == 1
        else:
            assert len(input_ids.shape) == 2

        if is_onehot:
            input_embeddings = self.onehot_ids_to_embedding(
                input_ids).unsqueeze(0)  # batch size 1
            _x = self.sentence_transformer(
                inputs_embeds=input_embeddings,
                attention_mask=mask.unsqueeze(0)
            )
        else:
            _x = self.sentence_transformer(
                input_ids=input_ids.unsqueeze(0),  # batch size 1
                attention_mask=mask.unsqueeze(0)
            )
        _x = self.mean_pooling(_x, mask)
        return _x

    def onehot_ids_to_embedding(self, onehot_ids):
        '''get input embeddings from onehot-encoded input_ids

        Args:
            onehot_ids: onehot-encoded input ids of shape (seq_len, vocab_size)

        Returns:
            input embeddings of shape (seq_len, dim)
        '''
        embeddings = self.sentence_transformer.embeddings
        # (vocab_size, dim)
        word_embeddings_mat = embeddings.word_embeddings.weight
        word_embeddings = onehot_ids.matmul(
            word_embeddings_mat)  # (seq_len, dim)

        seq_length = onehot_ids.size(0)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=onehot_ids.device)  # (seq_len,)
        position_embeddings = embeddings.position_embeddings(
            position_ids)  # (seq_len, dim)

        input_embed = word_embeddings + position_embeddings  # (seq_len, dim)
        input_embed = embeddings.LayerNorm(input_embed)  # (seq_len, dim)
        input_embed = embeddings.dropout(input_embed)  # (seq_len, dim)

        return input_embed

    @staticmethod
    def ids_to_onehot(ids, vocab_size):
        one_hot = torch.zeros((ids.shape) + (vocab_size,))
        assert(len(one_hot.shape) == 3)
        one_hot[
            torch.arange(one_hot.size(
                0)).unsqueeze(-1).repeat(1, one_hot.size(1)),
            torch.arange(one_hot.size(1)),
            ids
        ] = 1
        return one_hot.to(ids.device)

    @staticmethod
    def mean_pooling(x, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        args:
            x: sentence transformer last layer outputs of shape (1, batch_size, SEQ_LEN, HIDDEN_DIM_SIZE)

        returns:
            mean of last layer outpus across SEQ_LEN, taking attention mask into account
        '''
        token_embeddings = x[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
