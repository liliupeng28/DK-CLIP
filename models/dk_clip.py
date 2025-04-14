from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
from .msim import MultisnippetIntegrationTransformer
from .prompt import VideoSpecificPrompt, TextEncoder, PromptLearner
from .cfam import LocalCrossFrameCommunicationTransformer, CrossFrameCommunicationTransformerInteGration
from .temporal import Temporal_Transformer_Cls
from .text import class_descriptor_11
import sys
import warnings

sys.path.append("../")
from clip.model import CLIP, LayerNorm, Transformer
from clip import clip

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class DKCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 T=16,
                 droppath=0.,
                 msim_layers=1,
                 # prompt
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 N=4,
                 cfg=None
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.N = N
        # cross modal attention for textual prompt and visual feature
        self.prompts_generator_cmam = VideoSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha, )
        self.use_cache = use_cache
        # multi-snippet integration module
        self.msim = MultisnippetIntegrationTransformer(T=T // self.N, embed_dim=embed_dim, layers=msim_layers)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64

        # cross frame attention module in local snippet (through message tokens)
        self.visual = LocalCrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
            N=self.N
        )

        # cross frame visual semantic information calculation
        self.visual_semantic = CrossFrameCommunicationTransformerInteGration(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T // self.N,
            use_checkpoint=use_checkpoint,
            N=self.N
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # word embedding
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))

        self.initialize_parameters()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_local_image(self, image):
        return self.visual(image)

    def encode_visual_semantics(self, img_feature):
        return self.visual_semantic(img_feature)

    def encode_text(self, text):  # text encode
        x = self.token_embedding(text)

        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x
    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def encode_video(self, image):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)

        cls_features, img_features = self.encode_local_image(image)  # cls_features: class token; img_feature: image feature

        ##############
        cls_features = cls_features.view(b, t // self.N, self.N, -1).mean(dim=2, keepdim=False)
        img_features = img_features.view(b, t // self.N, self.N, -1, img_features.shape[-1]).mean(dim=2,keepdim=False)

        ######visual information modeling
        cls_featuresIG, img_featuresIG = self.encode_visual_semantics(img_features)
        img_featuresIG = self.prompts_visual_ln(img_featuresIG)
        img_featuresIG = img_featuresIG @ self.prompts_visual_proj
        cls_featuresIG = cls_featuresIG.view(b, t // self.N, -1)
        img_featuresIG = img_featuresIG.view(b, t // self.N, -1, cls_featuresIG.shape[-1])

        video_features, atts = self.msim(cls_features)
        return video_features, img_featuresIG, cls_features, atts


    def forward(self, image, text):
        logit_scale = self.logit_scale.exp()
        b = image.shape[0]
        video_features, img_features, cls_features, atts = self.encode_video(image)
        # global video feature
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        # local snippet feature
        cls_features = cls_features / cls_features.norm(dim=-1, keepdim=True)

        # noisy feature erasing based on visual similarity
        visual_smi = (torch.einsum("bd,bkd->bk", video_features, logit_scale * cls_features))
        visual_smi = normalize(visual_smi)
        erase_index = visual_smi.lt(0.15)
        img_features = cal_mean(img_features, erase_index)

        # enhancing textual prompts with visual contents
        if self.use_cache:
            text_features = self.cache_text(text)
        else:
            text_features = self.encode_text(text)
        text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        text_features = text_features + self.prompts_generator_cmam(text_features, img_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # global classification
        logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)
        # local classification
        logits_local = torch.einsum("bik,bjk->bij", cls_features, logit_scale * text_features)
        # fusion of local results based erase_index
        local_fuse = noise_erasing_fusion(logits_local, visual_smi, erase_index)

        return logits, local_fuse, video_features

# ablation study model
class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 T=8,
                 droppath=0.,
                 mit_layers=1,
                 # prompt
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 N=4,
                 cfg=None
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        # class_names = class_names_7
        class_descriptor = class_descriptor_11
        CLIP_model, _ = clip.loadclip(name='ViT-B/16',
                                  device='cpu')
        self.prompt_learner = PromptLearner(cfg, class_descriptor, CLIP_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(CLIP_model)

        self.image_encoder = CLIP_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=1,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.initialize_parameters()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def forward(self, image, text):

        n, t, c, h, w = image.shape
        image = image.contiguous().view(-1, c, h, w)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features.contiguous().view(n, t, -1)
        video_features = self.temporal_net(image_features)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.unsqueeze(0).expand(n, -1, -1)

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        output =  torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)#video_features @ text_features.t() / 0.01

        return output, output, video_features



def cal_mean(img_features, erase_index):
    # calculate mean of image feature based on erase_index
    b, n, c, h = img_features.size()
    mean = torch.zeros([b, c, h], dtype=torch.float32).cuda()
    for i in range(b):
        num = 0.0
        for j in range(n):
            if erase_index[i, j] == False:
                num += 1
                mean[i, :, :] = mean[i, :, :] + img_features[i, j, :, :]
        if num != 0:
            mean[i, :, :] = mean[i, :, :] / num
    return mean


def normalize(smi):
    b, n = smi.size()
    out_put = torch.zeros_like(smi)
    for i in range(b):
        out_put[i] = (smi[i] - torch.min(smi[i])) / (torch.max(smi[i]) - torch.min(smi[i]))

    return torch.softmax(out_put, dim=-1)


def noise_erasing_fusion(logits_local, visual_smi, erase_index):
    b, n, c = logits_local.size()
    output = torch.zeros([b, n, c], dtype=torch.float32).cuda()
    maks = visual_smi
    for i in range(b):
        for j in range(n):
            output[i, j, :] = logits_local[i, j, :] * (maks[i, j] * (1 - float(erase_index[i, j])))
    return output


def weight(logits_local, atts):
    b, n, c = logits_local.size()
    output = torch.zeros([b, n, c], dtype=torch.float32).cuda()
    for i in range(b):
        for j in range(n):
            output[i, j, :] = logits_local[i, j, :] * atts[i, j]
    return output


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1,
                prompts_layers=2, use_cache=True, msim_layers=4, N=4, cfg=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = DKCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        T=T, droppath=droppath, msim_layers=msim_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache, N=N, cfg=cfg
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    # logger.info(f"load pretrained CLIP: {msg}")

    return model.eval()


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, prompts_alpha=1e-1,
         prompts_layers=2, msim_layers=1, N=4, cfg=None
         ):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath,
                        use_checkpoint=use_checkpoint, logger=logger,
                        prompts_alpha=prompts_alpha,
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        msim_layers=msim_layers,
                        N=N,
                        cfg=cfg
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()
