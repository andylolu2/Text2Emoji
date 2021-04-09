import pandas as pd

from config import DreamerConfig


class EmojiDict():
    df = pd.read_csv(
        f'{DreamerConfig.PROJECT_ROOT}/{DreamerConfig.SINGLE_VOCAB_PATH}')

    def __init__(self, tfidf_threshold=0.75):
        self.vocab = self.df['feature'][self.df['tfidf_score']
                                        > tfidf_threshold].to_list()
        self.feature_to_emoji = {}
        self.emoji_to_feature = {}

        for i, row in self.df.iterrows():
            self.feature_to_emoji[row['feature']] = row['emoji']
            self.emoji_to_feature[row['emoji']] = row['feature']

    def sentence_to_emoji(self, sentence):
        return [self.str_to_emoji(emoji_str) for emoji_str in sentence]

    def str_to_emoji(self, emoji_str):
        try:
            return self.feature_to_emoji[emoji_str]
        except KeyError:
            raise KeyError(f"emoji string: {emoji_str} not found")

    def emoji_to_str(self, emoji):
        try:
            return self.emoji_to_feature[emoji]
        except KeyError:
            raise KeyError(f"emoji: {emoji} not found")
