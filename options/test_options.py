from .base_options import BaseOptions
from .train_options import str2bool

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')

        self.parser.add_argument('--choice_txt_img', type=str2bool, default=False, help='use CLIP text embeddings(False) or img embeddings')
        self.parser.add_argument('--use_PCA', type=str2bool, default=False, help='use PCA to reduce the dimension of CLIP text embeddings')
        self.parser.add_argument('--prior_type', type= str, default= 'MLP', help='Type for the prior network, there are MLP..., None means we do not want a prior network')
        self.parser.add_argument('--use_CLIP_img_txt_loss', type=str2bool, default=False, help='Use CLIP image and text loss')
        self.isTrain = False
