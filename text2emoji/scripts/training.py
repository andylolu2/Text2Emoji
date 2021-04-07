import traceback
import math
import os
import logging
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm

from text2emoji.classes.dataset import Text2EmojiDataSet
from text2emoji.classes.model import Text2EmojiModel
from config import TrainConfig

os.system(f"rm -rf {TrainConfig.TENSORBOARD_LOGDIR}/*")

logging.root.setLevel(logging.INFO)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.set_printoptions(threshold=100)  # for printing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(
    log_dir=f"{TrainConfig.TENSORBOARD_LOGDIR}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/")

# Set up dataset
dset = Text2EmojiDataSet(size=100)
train_set, val_set = torch.utils.data.random_split(
    dset,
    lengths=[
        math.ceil(len(dset) * (1 - TrainConfig.VAL_SPLIT)),
        math.floor(len(dset) * TrainConfig.VAL_SPLIT)
    ]
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=TrainConfig.BATCH_SIZE,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=TrainConfig.BATCH_SIZE,
    shuffle=True
)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Training on device: {device}")

# Setup model
emoji_vocab = pd.read_csv(TrainConfig.EMOJI_VOCAB_PATH, usecols=[
                          'cleaned_name'])['cleaned_name'].to_list()
model = Text2EmojiModel(emoji_vocab, device).to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=TrainConfig.LEARNING_RATE,
    momentum=TrainConfig.SGD_MOMENTUM)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=TrainConfig.SCHEDULER_STEP_SIZE,
    gamma=TrainConfig.SCHEDULER_GAMMA)

# mask, x, y = next(iter(train_loader))
# print(mask.shape)  # (BATCH_SIZE, SEQ_LEN)
# print(x.shape)  # (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
# print(torch.transpose(x, 0, 1).shape)  # (SEQ_LEN, BATCH_SIZE, HIDDEN_DIM)
# print(y.shape)  # (BATCH_SIZE, HIDDEN_DIM)
print('Training...', flush=True)
print('Please view on tensorboard...', flush=True)
rewards_history = []
for epoch in tqdm(
        range(TrainConfig.EPOCHS),
        bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):

    writer.add_scalar('Train/learning rate',
                      scheduler.get_last_lr()[0],
                      epoch * len(train_loader))

    running_loss = .0
    running_setence_reward = .0
    running_length_reward = .0
    running_emoji_length = .0
    for i, data in enumerate(train_loader, 0):
        body, mask, x, y = data
        mask, x, y = mask.to(device), x.to(device), y.to(device)
        x = torch.transpose(x, 0, 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x, mask)
        y_pred, log_prob, emojis = model.embeddings_to_emojis_to_sentence_embeddings(
            outputs, sample_size=TrainConfig.SAMPLE_SIZE)
        y_real = y.repeat(TrainConfig.SAMPLE_SIZE, 1, 1).transpose(0, 1)
        sentence_reward = -torch.mean((y_pred - y_real) ** 2, -1)
        emoji_lengths = [[len(sample) for sample in batch] for batch in emojis]
        length_reward = -torch.tensor(emoji_lengths).to(device) * \
            TrainConfig.SHORT_EMOJI_REWARD
        reward = sentence_reward + length_reward
        expected_reward = np.average(rewards_history) if len(
            rewards_history) > 0 else 0
        # print(expected_reward)
        loss = torch.mean(-log_prob * (reward - expected_reward))
        # if epoch > 100:
        #     print(loss)
        #     print(log_prob)
        #     print(sentence_reward)
        #     print(length_reward)
        #     raise Exception()
        # print(next(model.parameters()))
        loss.backward()
        # for param in model.parameters():
        #     print(param.grad)
        # print("log prob: ")
        # print(log_prob.grad)
        optimizer.step()

        rewards_history.append(torch.mean(reward).detach().cpu().numpy())
        rewards_history = rewards_history[-TrainConfig.REWARD_HISTORY_LEN:]
        running_loss += loss.item()
        running_setence_reward += torch.mean(
            sentence_reward.float()).detach().cpu().numpy()
        running_length_reward += torch.mean(
            length_reward.float()).detach().cpu().numpy()
        running_emoji_length += sum([sum(batch) for batch in emoji_lengths]) / (
            TrainConfig.BATCH_SIZE * TrainConfig.SAMPLE_SIZE)
        if i % TrainConfig.BATCH_PER_LOG == TrainConfig.BATCH_PER_LOG - 1:
            # Log the running loss
            mean_loss = running_loss / TrainConfig.BATCH_PER_LOG
            writer.add_scalar('Train/loss',
                              mean_loss,
                              epoch * len(train_loader) + i)
            writer.add_scalar('Train/emoji length',
                              running_emoji_length / TrainConfig.BATCH_PER_LOG,
                              epoch * len(train_loader) + i)
            writer.add_scalar('Train/expected reward',
                              np.average(rewards_history),
                              epoch * len(train_loader) + i)
            writer.add_scalar('Train/sentence reward',
                              running_setence_reward / TrainConfig.BATCH_PER_LOG,
                              epoch * len(train_loader) + i)
            writer.add_scalar('Train/length reward',
                              running_length_reward / TrainConfig.BATCH_PER_LOG,
                              epoch * len(train_loader) + i)
            writer.add_text('Example outputs',
                            f'Body: {body[0]}\nOutput: {" ".join(emojis[0][0])}',
                            epoch * len(train_loader) + i)
            with torch.no_grad():
                val_loss = .0
                val_emoji_len = .0
                for body_val, mask_val, x_val, y_val in val_loader:
                    mask_val = mask_val.to(device)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    x_val = torch.transpose(x_val, 0, 1)
                    outputs_val = model(x_val, mask_val)
                    y_pred_val, log_prob_val, emojis_val = model.embeddings_to_emojis_to_sentence_embeddings(
                        outputs_val, debug=False)
                    reward_val = -torch.mean((y_pred_val - y_val) ** 2, -1)
                    loss_val = torch.mean(log_prob_val * (reward_val))
                    val_loss += loss_val.item()
                    val_emoji_len += np.average([[len(sample)
                                                for sample in batch] for batch in emojis])
                writer.add_scalar(
                    'Validation/loss',
                    val_loss / len(val_loader),
                    epoch * len(train_loader) + i)
                writer.add_scalar(
                    'Validation/emoji length',
                    val_emoji_len / len(val_loader),
                    epoch * len(train_loader) + i)
            running_loss = .0
            running_setence_reward = .0
            running_length_reward = .0
            running_emoji_length = .0

    scheduler.step()
print("Done!")
