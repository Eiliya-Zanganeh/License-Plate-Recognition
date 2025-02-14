import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

from Recogniser_Module.utils import CTCLabelConverter, AttnLabelConverter
from Recogniser_Module.model import Model
from torchvision import transforms
from config import *


class Recogniser:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cudnn.benchmark = True
        cudnn.deterministic = True
        self.num_gpu = torch.cuda.device_count()
        self.__load_model()

    def __load_model(self):
        """ model configuration """
        if 'CTC' in RECOGNISER_PREDICTION:
            self.converter = CTCLabelConverter(RECOGNISER_CHARACTER)
        else:
            self.converter = AttnLabelConverter(RECOGNISER_CHARACTER)
        num_class = len(self.converter.character)
        self.model = Model(num_class)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        self.model.load_state_dict(torch.load(RECOGNISER_SAVED_MODEL, map_location=self.device))

    def __call__(self, image_array):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((RECOGNISER_IMGH, RECOGNISER_IMGW)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image_array).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            batch_size = image_tensor.size(0)
            length_for_pred = torch.IntTensor([RECOGNISER_BATCH_MAX_LENGTH] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, RECOGNISER_BATCH_MAX_LENGTH + 1).fill_(0).to(self.device)

            if 'CTC' in RECOGNISER_PREDICTION:
                preds = self.model(image_tensor, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, preds_size)
            else:
                preds = self.model(image_tensor, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            pred = preds_str[0]
            if 'Attn' in RECOGNISER_PREDICTION:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                preds_max_prob = preds_max_prob[0, :pred_EOS]
            else:
                preds_max_prob = preds_max_prob[0]

            confidence_score = preds_max_prob.cumprod(dim=0)[-1].item()
            print(f'Predicted: {pred}, Confidence Score: {confidence_score:.4f}')
            return pred, confidence_score
