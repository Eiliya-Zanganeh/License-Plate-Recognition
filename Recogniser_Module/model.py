import torch.nn as nn

from Recogniser_Module.modules.transformation import TPS_SpatialTransformerNetwork
from Recogniser_Module.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, \
    ResNet_FeatureExtractor
from Recogniser_Module.modules.sequence_modeling import BidirectionalLSTM
from Recogniser_Module.modules.prediction import Attention
from config import *


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.stages = {
            'Trans': RECOGNISER_TRANSFORMATION,
            'Feat': RECOGNISER_FEATUREEXTRACTION,
            'Seq': RECOGNISER_SEQUENCEMODELING,
            'Pred': RECOGNISER_PREDICTION
        }

        """ Transformation """
        if RECOGNISER_TRANSFORMATION == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=RECOGNISER_NUM_FIDUCIAL,
                I_size=(RECOGNISER_IMGH, RECOGNISER_IMGW),
                I_r_size=(RECOGNISER_IMGH, RECOGNISER_IMGW),
                I_channel_num=RECOGNISER_INPUT_CHANNEL
            )
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if RECOGNISER_FEATUREEXTRACTION == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(RECOGNISER_INPUT_CHANNEL, RECOGNISER_OUTPUT_CHANNEL)
        elif RECOGNISER_FEATUREEXTRACTION == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(RECOGNISER_INPUT_CHANNEL, RECOGNISER_OUTPUT_CHANNEL)
        elif RECOGNISER_FEATUREEXTRACTION == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(RECOGNISER_INPUT_CHANNEL, RECOGNISER_OUTPUT_CHANNEL)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = RECOGNISER_OUTPUT_CHANNEL  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if RECOGNISER_SEQUENCEMODELING == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, RECOGNISER_HIDDEN_SIZE, RECOGNISER_HIDDEN_SIZE),
                BidirectionalLSTM(RECOGNISER_HIDDEN_SIZE, RECOGNISER_HIDDEN_SIZE, RECOGNISER_HIDDEN_SIZE))
            self.SequenceModeling_output = RECOGNISER_HIDDEN_SIZE
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if RECOGNISER_PREDICTION == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_classes)
        elif RECOGNISER_PREDICTION == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, RECOGNISER_HIDDEN_SIZE, num_classes)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                         batch_max_length=RECOGNISER_BATCH_MAX_LENGTH)

        return prediction
