import os


class BaseConfig():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # Path to ONNX compiled distilbert for fast inference
    DISTILBERT_ONNX_PATH = "models/onnx/distilbert/model.onnx"
    # Path to ONNX compiled sentence transformer for fast inference
    SENTENCE_ONNX_PATH = "models/onnx/sentence-transformer/model.onnx"
    # Path to reddit text training data for RL approach
    H5DSET_PATH = "processed_data/nn_data.hdf5"
    # Path to reddit text testing data for RL approach
    H5DSET_TEST_PATH = "processed_data/nn_test_data.hdf5"
    # Path to processed emojis
    EMOJI_VOCAB_PATH = "processed_data/emoji_vocab.csv"
    # Huggingface model hub reference name for distilbert
    DISTILBERT_NAME = "distilbert-base-uncased"
    # Huggingface model hub reference name for sentence transformer
    SENTENCE_TRANSFORMER_NAME = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"

    # Cannot be instantiated, must be inherited
    def __new__(cls, *args, **kwargs):
        if cls is BaseConfig:
            raise TypeError("Base Config may not be instantiated")
        return object.__new__(cls, *args, **kwargs)


class PrepConfig(BaseConfig):
    # Batch size for processing reddit dataset
    BATCH_SIZE = 1
    # Maximum length for reddit message before truncating
    MAX_LEN = 64
    # The proportion of english characters required to use
    # it as training / testing data
    ENGLISH_THRESHOLD = 0.9
    # Sample text for debuggnig
    SAMPLE_TEXT = "Are you really implying we return to those times or anywhere near that political environment?  If so, you won't have much luck selling the American people on that governance concept without ushering in American Revolution 2.0."
    # The chunk size to process the reddit data set in
    PD_CHUNK_SIZE = 1
    # The proportion of data used as testing data (Not for validation)
    TESTING_SPLIT = 0.1
    # The rounding decimal place of the no. of testing data
    TESTING_SPLIT_ROUND = -3  # Round to nearest thousands


class TrainConfig(BaseConfig):
    # The End of Sentence special token for distilbert
    # and sentence transformer
    EOS_TOKEN = '[EOS]'
    # The log directory for tensorboard writer
    TENSORBOARD_LOGDIR = 'logs/runs'
    # The maximum sequence length to process before truncation
    MAX_EMOJI_SENTENCE_LEN = 64
    # The proportion of data to use as validation data
    VAL_SPLIT = 0.1
    # Batch size to perform policy gradient
    BATCH_SIZE = 8
    # No. of samples to draw from emoji-sentence prob distribution
    SAMPLE_SIZE = 1
    # Learning for optimizer
    LEARNING_RATE = 1e-3
    # Step size for learning rate scheduler
    SCHEDULER_STEP_SIZE = 5
    # Gamma value for learning rate scheduler
    SCHEDULER_GAMMA = 0.5
    # Momemtum factor for SGD optimzer
    SGD_MOMENTUM = 0.95
    # No. of epochs to train for
    EPOCHS = 10
    # No. of batches per logging to tensorboard
    BATCH_PER_LOG = 4
    # No. of reward history data points to store
    REWARD_HISTORY_LEN = 10
    # The reward factor for giving shorter emoji-sentences
    SHORT_EMOJI_REWARD = 1e-4


class DreamerConfig(BaseConfig):
    # The process emojis for DeepDream
    SINGLE_VOCAB_PATH = "processed_data/emoji_vocab_single.csv"
    # Output destination of the dreaming resutls, relative to project root
    OUPUT_FILE = "/text2emoji_dream/outputs/dreams.txt"
    # No. of epochs to train for
    EPOCHS = 4000
    # No. of epocjs per printing to console
    EPOCH_PER_LOG = 100
    # Initial learning rate of optimizer
    OPTIMIZER_INIT_LR = 0.4
    # Gamma for learning rate scheduler
    SCHEDULER_GAMMA = 0.25
    # Miletones for learning rate scheduler
    SCHEDULER_MILETONES = [2500, 3500, 4500]
    # The final sharpness factor for Sigmoid for smoothed
    # attention mask (Not used anymore)
    SIGMOID_END_SHARPNESS = 1.5
    # The starting sharpness factor for Sigmoid for smoothed
    # attention mask (Not used anymore)
    SIGMOID_START_SHARPNESS = 1.5
    # The starting temperature of smooth one-hot encoed ids
    SM_START_TEMPERATURE = 2
    # The ending temperature of sharpened smooth one-hot encoed ids
    SM_END_TEMPERATURE = 0.5
