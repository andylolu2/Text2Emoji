# Building a :abcd: to :grin: sequence translator

## Problem

I wanted to build a model that can translate text to emojis. 

For example, 

`I drove to the forest with a waterfall, took a picture and photoshopped it on my computer` 

might be represented as 

:car::herb::evergreen_tree::national_park::droplet::cloud_with_rain::camera::computer::artist::floppy_disk: 

As you could see, the task is quite non-trivial. (The above 
translation is inspired by the model)

## Approaches

### **TL;DR**

A few methods were tried. In the end, *[DeepDreaming on sentence transformer](#:three:<sup>rd</sup>-attempt:-DeepDreaming-on-Sentence-Transformer)* gave the best results.

Main script in [text2emoji_dream/scripts/dream.py](text2emoji_dream/scripts/dream.py).

```console
(.venv) $ python -m text2emoji_dream.scripts.dream
```

[Fine-tuning](#onest-attempt-fine-tuning-distilbert) |
[Fine-tuning with RL](#twond-attempt-fine-tuning-distlibert-with-rl) |
[DeepDream](#threerd-attempt-deepdreaming-on-sentence-transformer)

Sample Results:

```text
Input: photoshop, 2
Output: 📷🧑‍🎨  (Camera Artist)


Input: Only in darkness can you see the stars, 5
Output: 👀⌚🥸⭐🧗‍♀️  (Eye Watch Disguise Star Climbing)


Input: Butterfly, 1
Output: 🦋  (Butterfly)


Input: Translate text to emojis using deep learning, 5
Output: ✍️⛰️ℹ️🔗🧠  (Writing Mountain Information Link Brain)


Input: The caterpillar only takes 2 weeks 
to grwo into a butterfly, 6
Output: 🇱🇦🔅👪🥚🕑🦋  (Laos ​Dim Family Egg Two️ Butterfly)


Input: The Great Barrier Reef in Australia is one the 
most beautiful places on Earth, 7
Output: 🔑🤴🇱🇨💙🇦🇺🔝🧑  
(Key Prince Lucia Blue Australia Top Adult)
```

### :one:<sup>st</sup> attempt: Fine-tuning DistilBert

Uses: [Pretained DistilBert](https://huggingface.co/distilbert-base-uncased), 
[Pretrained Sentence Transformer](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens),
[Reddit Dataset](https://www.kaggle.com/reddit/reddit-comments-may-2015)

#### Method

*Note: this method doesn't actually work. See next section*

1. Take a message `input` from the Reddit Dataset. 
Gets its sentence embedding using the Sentence Transformer.
This is our `target`.

2. Build a token classificatioin head on top of pretrained 
DistilBert to output of sequence of token probabilities, 
where each token is an emoji. Feed the same `input` into 
this network and get a `emoji_sequence`.

3. Then, transform the `emoji_sequence` into raw 
string (`emoji_sentence`) by taking the standardized 
text representation of each emoji. For example, 
:cloud_with_rain: has the text 
representation of *cloud with rain*.

4. Feed this emoji-sentence into the Sentence Transsformer 
and get its sentence embedding. This is our `output`. In a 
perfect world, where `emoji_sequence` fully encapsulates 
the meaning of `input`, `output` and `target` should be 
quite similar. 

5. We train the classification head while keeping other 
parameterse frozen to minimize the distance between 
`output` and `target` for all texts.

#### Problem

Step 3 of the method (where the `emoji_sequence` is transformed 
into a raw string) is actually non-differentiable. This 
means that we cannot directly train the parameters of the 
classification head with supervised learning since 
backpropagation is not possible. 

Step 3 is necessary since (not all) emojis are not in the vocab list 
of the Sentence Transformer, plus inputs to the Sentence 
Transformer requires some special tokens at specific 
positions. As a result, we cannot directly 
go from `emoji_sequence` to inputs of the Sentence Transformer 
without doing tokenization. (Which is non-differentiable)

This problem led to the method of the 2<sup>nd</sup> attempt.

### :two:<sup>nd</sup> attempt: Fine-tuning DistliBert with RL

Uses: [Pretained DistilBert](https://huggingface.co/distilbert-base-uncased), 
[Pretrained Sentence Transformer](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens),
[Reddit Dataset](https://www.kaggle.com/reddit/reddit-comments-may-2015)

#### Method

From the 1<sup>st</sup> attempt, the main problem was the 
non-differentiablility. To overcome this, Reinforcement 
Learning (RL) algorithms might be useful, since they 
do not need to know how exactly are the reward (loss) is 
calculated in the environment. 

Putting our setup into a RL setting:

- Environment: Sentence Transformer
- Agent: DistilBert with classification head
- State: Message from Reddit Dataset
- Action: `emoji_sequence`
- Reward: similarity between `output` and `target`

In particular, given some state (text), we want to find the 
optimal action (emoji sequence), that can maximize the 
reward (similarity between `output` and `target`)

The Policy Gradient algorithm with Advantage was implemented 
to solve this problem.

#### Problem

The optimization algorithm simply does not work very well 
where it gets stuck in local optima very often. 

I do not understand RL algorithms in-depth so I do not
really have an intuition on where exactly the problem 
lies in.

### :three:<sup>rd</sup> attempt: DeepDreaming on Sentence Transformer

Uses: [Pretrained Sentence Transformer](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens)

#### Method

Instead of going through a DistilBert model to predict 
the text, we directly optimze the inputs to the 
sentence transformer so that the final sentence 
embedding is similar to `target`.

To ensure that the inputs that we are optimizing always 
end with in valid emoji sequences, few sacrifices were 
made:

1. Only emojis that can be translated into one specific 
token in the Sentence Transformer's 
vocab list are considered. This means that for each 
emoji's text representation, only 1 word can be selected 
to represent that emoji and (hopefully) that word will 
be in the Sentence Transformer's vocab list. If it is 
not in the vocab list, we simply discard that emoji 
and pretend that emoji never existed :wink: 
(This is the case for about 20% of emojis). For most 
emojis, this step is trivial since their text representation 
only consists of 1 word.

2. To select one word in the emoji's text representation 
to represent that emoji, the **tf-idf** algorithm is used 
to find the most important token in the string. 

Unfortunately, naively applying the DeepDream algorithm to 
text inputs does not work. This is because a sequence of 
tokens is discrete and is non-differentiable. To overcome this 
problem, the one-hot encoding vectors for the input tokens 
are transformed into a soft-max vector that is not fixed to 
0 or 1. 

To ensure that the final soft-max vectors somewhat resembles 
a one-hot vector, an annealing temperature `t` 🠒 0 is used, 
where the inputs to be optimized are divided by `t` before 
being soft-maxed. This results to a more and more spiky 
probability distribution as `t` 🠒 0 during training.

Vocabs that are not emojis are masked.

#### Problem

This still seems to be quite a difficult problem and the 
outputs are not very compelling. One of the problems is 
that the Sentence Transformer expects some specific formats 
of special tokens as inputs and that is difficult to learn.

To reduce the difficulty of the problem, the final length 
of the output is given to the model as an additional 
parameter so that the format of the inputs can be pre-defined.

#### Results

These are some sample results with user-specified lengths:

```text
Input: photoshop, 2
Output: 📷🧑‍🎨  (Camera Artist)


Input: Only in darkness can you see the stars, 5
Output: 👀⌚🥸⭐🧗‍♀️  (Eye Watch Disguise Star Climbing)


Input: Butterfly, 1
Output: 🦋  (Butterfly)


Input: Translate text to emojis using deep learning, 5
Output: ✍⛰️ℹ️🔗🧠  (Witing Mountain Information Link Brain)


Input: The caterpillar only takes 2 weeks 
to grwo into a butterfly, 6
Output: 🇱🇦🔅👪🥚🕑🦋  (Laos ​Dim Family Egg Two️ Butterfly)


Input: The Great Barrier Reef in Australia is one the 
most beautiful places on Earth, 7
Output: 🔑🤴🇱🇨💙🇦🇺🔝🧑  
(Key Prince Lucia Blue Australia Top Adult)
```

You can see that sometimes, the model does make some quite 
remarkable / creative expressions. My favourite is definitely 
":artist::camera:" for "*photoshop*". 

However, at other times, the appearance of some emojis 
cannot be explained very intuitively, such as the connection 
between ":mountain:" and "*Translate text to emojis using deep learning*".

#### Limitations

1. Fixed translation length

2. Requires tuning of hyperparameters depending on 
input. For example, it is observed that sequences of 
longer length require more epochs / higher learning rate.

3. Takes a *LONG* time for inference. Each input takes about 
:one: minute to compute on a GPU.

4. Works ok for shorter inputs and outputs. More noisy 
emojis are introduced for logner sequences.

#### Analysis

There are 2 main possible sources of error:

1. Lack of robustness of the Sentence Transformer

    - The Sentence Transformer is trained on Wikipedia
    and Stanford SNLI corpus, which is not *that* messy.
    As a result, it might not be robust enough 
    to handle such weirdly-worded emoji-sentences. 
    Thus, the optimizer might be able to exploit 
    such knowledge "holes", where similar outputs 
    don't imply similar inputs.

    - This can maybe be alleviated by another Sentence 
    Transformer that is trained independently. The two 
    Sentence Transformers might have different "holes" in 
    their knowledge where they can supplement each other.

2. Poor optimization of inputs

    - Another possiblity is that the optimizer simply 
    cannot find the global optima. 

    - Can maybe spend more time trying different loss functions, 
    optimizers, hyperparameters... etc.



## Datasets

- [Emoji dictionary](https://www.kaggle.com/eliasdabbas/emoji-data-descriptions-codepoints)

- [Reddit dataset](https://www.kaggle.com/reddit/reddit-comments-may-2015)
    - Number of entries: 54,504,410
