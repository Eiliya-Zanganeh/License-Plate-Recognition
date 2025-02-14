import string

################### Database config ###################

DATABASE_PATH = 'plate.db'

################### Verification config ###############

VERIFICATION_THRESHOLD = .8

################### Detector config ###################

DETECTOR_WEIGHT = './Detector_Module/detector_weight.pt'
DETECTOR_THRESHOLD = .1

################## Recogniser config ###################

RECOGNISER_WORKERS = 4  # number of data loading workers
RECOGNISER_BATCH_SIZE = 192  # input batch size
RECOGNISER_SAVED_MODEL = './Recogniser_Module/recogniser_weight.pth'  # path to saved_model to evaluation
RECOGNISER_THRESHOLD = .5

# Data processing

RECOGNISER_BATCH_MAX_LENGTH = 25  # maximum-label-length
RECOGNISER_IMGH = 32  # the height of the input image
RECOGNISER_IMGW = 100  # the width of the input image
RECOGNISER_RGB = None  # use rgb input
RECOGNISER_CHARACTER = '0123456789abcdefghijklmnopqrstuvwxyz'  # character label
RECOGNISER_SENSITIVE = None  # for sensitive character mode
RECOGNISER_PAD = None  # whether to keep ratio then pad for image resize

# Model Architecture

RECOGNISER_TRANSFORMATION = 'TPS'  # Transformation stage. None|TPS
RECOGNISER_FEATUREEXTRACTION = 'ResNet'  # FeatureExtraction stage. VGG|RCNN|ResNet
RECOGNISER_SEQUENCEMODELING = 'BiLSTM'  # SequenceModeling stage. None|BiLSTM
RECOGNISER_PREDICTION = 'Attn'  # Prediction stage. CTC|Attn
RECOGNISER_NUM_FIDUCIAL = 20  # number of fiducial points of TPS-STN
RECOGNISER_INPUT_CHANNEL = 1  # the number of input channel of Feature extractor
RECOGNISER_OUTPUT_CHANNEL = 512  # the number of output channel of Feature extractor
RECOGNISER_HIDDEN_SIZE = 256  # the size of the LSTM hidden state

########################################################

if RECOGNISER_SENSITIVE:
    RECOGNISER_CHARACTER = string.printable[:-6]  # same with ASTER setting (use 94 char).

if RECOGNISER_RGB:
    RECOGNISER_INPUT_CHANNEL = 3
