class BaseConfig():
    PROJECT_ROOT = "/home/andylo/Projects/Text2Emoji"
    DISTILBERT_ONNX_PATH = "models/onnx/distilbert/model.onnx"
    SENTENCE_ONNX_PATH = "models/onnx/sentence-transformer/model.onnx"
    H5DSET_PATH = "processed_data/nn_data.hdf5"
    H5DSET_TEST_PATH = "processed_data/nn_test_data.hdf5"
    EMOJI_VOCAB_PATH = "processed_data/emoji_vocab.csv"
    DISTILBERT_NAME = "distilbert-base-uncased"
    SENTENCE_TRANSFORMER_NAME = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"

    def __new__(cls, *args, **kwargs):
        if cls is BaseConfig:
            raise TypeError("Base Config may not be instantiated")
        return object.__new__(cls, *args, **kwargs)


class PrepConfig(BaseConfig):
    BATCH_SIZE = 1
    MAX_LEN = 64
    ENGLISH_THRESHOLD = 0.9
    SAMPLE_TEXT = "Are you really implying we return to those times or anywhere near that political environment?  If so, you won't have much luck selling the American people on that governance concept without ushering in American Revolution 2.0."
    PD_CHUNK_SIZE = 1
    TESTING_SPLIT = 0
    TESTING_SPLIT_ROUND = -3  # Round to nearest thousands


class TrainConfig(BaseConfig):
    EOS_TOKEN = '[EOS]'
    TENSORBOARD_LOGDIR = 'logs/runs'
    MAX_EMOJI_SENTENCE_LEN = 64
    VAL_SPLIT = 0.1
    BATCH_SIZE = 8
    SAMPLE_SIZE = 1
    LEARNING_RATE = 1e-3
    SCHEDULER_STEP_SIZE = 5
    SCHEDULER_GAMMA = 0.5
    SGD_MOMENTUM = 0.95
    EPOCHS = 10
    BATCH_PER_LOG = 4
    REWARD_HISTORY_LEN = 10
    SHORT_EMOJI_REWARD = 1e-4


class DreamerConfig(BaseConfig):
    SINGLE_VOCAB_PATH = "processed_data/emoji_vocab_single.csv"
    EPOCHS = 4000
    EPOCH_PER_LOG = 100
    SIGMOID_END_SHARPNESS = 1.5
    SIGMOID_START_SHARPNESS = 1.5
    SM_START_TEMPERATURE = 2
    SM_END_TEMPERATURE = 0.5
