import numpy as np
import pandas as pd
import torch
import transformers

from text2emoji_dream.classes.Text2EmojiDreamer import Text2EmojiDreamer
from text2emoji_dream.classes.EmojiDict import EmojiDict
from config import DreamerConfig


def dream(text, guess_len, device='cuda'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        DreamerConfig.SENTENCE_TRANSFORMER_NAME)
    model = Text2EmojiDreamer(device).to(device)
    temperatures = np.geomspace(
        DreamerConfig.SM_START_TEMPERATURE,
        DreamerConfig.SM_END_TEMPERATURE,
        DreamerConfig.EPOCHS)
    # sharpnesses = np.linspace(
    #     DreamerConfig.SIGMOID_START_SHARPNESS,
    #     DreamerConfig.SIGMOID_END_SHARPNESS,
    #     DreamerConfig.EPOCHS
    # )

    emoji_dict = EmojiDict(tfidf_threshold=0.8)
    vocabs = emoji_dict.vocab
    for vocab in vocabs:
        assert vocab in tokenizer.get_vocab()
    vocab_ids = tokenizer.convert_tokens_to_ids(vocabs)
    assert len(vocab_ids) == len(set(vocab_ids))

    tokenized = tokenizer(text, return_tensors='pt',
                          padding='max_length', truncation=True)
    before_fixed = torch.tensor(
        [tokenizer.convert_tokens_to_ids('[CLS]')], device=device)
    modifiable = torch.zeros(
        guess_len, device=device)  # place holder
    after_fixed = torch.tensor(
        [tokenizer.convert_tokens_to_ids('[SEP]')], device=device)
    input_ids = torch.cat(
        (before_fixed, modifiable, after_fixed)).type(torch.long)

    onehot_ids = model.ids_to_onehot(
        input_ids.unsqueeze(0), tokenizer.vocab_size)[0]
    before_fixed_oh = onehot_ids[:1]
    modifiable_oh = torch.rand_like(
        onehot_ids[1:-1], requires_grad=True, device=device)
    after_fixed_oh = onehot_ids[-1:]

    # Mask tokens that are not emojis
    vocab_mask = torch.zeros_like(modifiable_oh, device=device)
    vocab_mask[:, vocab_ids] = 1

    # mask_index = torch.tensor(
    #     0,
    #     dtype=torch.float32,
    #     requires_grad=True,
    #     device=device)

    optimizer = torch.optim.AdamW(
        [modifiable_oh], lr=1e-1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2500, gamma=0.2)
    target = model(tokenized.input_ids[0].to(device),
                   tokenized.attention_mask[0].to(device), is_onehot=False)

    for i in range(DreamerConfig.EPOCHS):
        temperature = temperatures[i]
        # sharpness = sharpnesses[i]

        sm_modify = torch.softmax(
            (modifiable_oh * vocab_mask) / temperature, dim=-1)
        fused_oh = torch.cat(
            (before_fixed_oh, sm_modify, after_fixed_oh), dim=0)

        # mask_prop = torch.sigmoid(mask_index)  # proportion of text the mask
        # # soft attention cut-off (to change input length)
        # mask = torch.sigmoid(
        #     -(torch.arange(tokenizer.model_max_length - 1, device=device) - tokenizer.model_max_length * mask_prop) * sharpness)
        # # soft attention mask
        # fused_mask = torch.cat((torch.tensor([1], device=device), mask), dim=0)
        fused_mask = torch.ones(fused_oh.size(0), device=device)

        optimizer.zero_grad()
        output = model(fused_oh, fused_mask, is_onehot=True)
        loss = loss_fn(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if i % DreamerConfig.EPOCH_PER_LOG == DreamerConfig.EPOCH_PER_LOG - 1:
            print(
                f'step: {i} loss: {loss.item()}')
            final_ids = torch.argmax(fused_oh, dim=-1)
            # final_index = int(mask_prop * tokenizer.model_max_length) + 1
            # print(final_ids)
            # print(final_index)

            print(tokenizer.convert_ids_to_tokens(final_ids))

        if i == DreamerConfig.EPOCHS - 1:
            sentence = tokenizer.convert_ids_to_tokens(final_ids)[1:-1]
            print(emoji_dict.sentence_to_emoji(sentence))


if __name__ == '__main__':
    text = "photoshop"
    guess_len = 2

    dream(text, guess_len)
