"""FER2013 seven-class ensemble inference from a cropped face on CPU."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT/'strategy'))
sys.path.append(str(ROOT/'candidate_code'))
from combine import predict as combine_predictions
from source_models import AffectLens, OxfordAlexNet, OxfordVGGM
from candidate_infer import fuse, run

CLASSES_ZH = ['愤怒','厌恶','恐惧','高兴','悲伤','惊讶','中性']


class CpuEnsemble:
    """Load the exact model family, order and fusion policy used for evaluation."""

    def __init__(self, policy_path, weights, threads=1):
        import onnxruntime as ort
        import timm
        from torchvision.models import resnet34

        self.policy = json.loads(Path(policy_path).read_text())
        self.names = [model['name'] for model in self.policy['models']]
        self.models = {}
        weights = Path(weights)
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        for name in self.names:
            if name == 'tyom':
                checkpoint = torch.load(weights/'tyom_fer2013_emotion_resnet18_v2.pt',
                                        map_location='cpu',weights_only=True)
                model = timm.create_model(checkpoint['model_name'],pretrained=False,num_classes=7)
                model.load_state_dict(checkpoint['state_dict'],strict=True)
                model = model.eval().to(memory_format=torch.channels_last)
                fuse(model)
            elif name in {'affectlens','affectlens_rehead'}:
                model = AffectLens(weights/'affectlens.pt',threads)
                if name == 'affectlens_rehead':
                    with np.load(ROOT/'frozen_head.npz',allow_pickle=False) as head:
                        if str(head['feature_kind']) != 'embedding':
                            raise ValueError('Unexpected AffectLens feature kind')
                        self.head_weight, self.head_bias = head['weight'], head['bias']
            elif name == 'oxford':
                model = OxfordAlexNet(weights/'oxford_alexnet.pth',threads)
            elif name == 'vggm':
                model = OxfordVGGM(weights/'oxford_vggm.pth',threads)
            elif name in {'densenet','cbam'}:
                options = ort.SessionOptions()
                options.intra_op_num_threads=threads
                options.inter_op_num_threads=1
                model = ort.InferenceSession(str(weights/('densenet121_224_int8.onnx' if name == 'densenet' else 'cbam_resnet50_224_int8.onnx')),
                                              sess_options=options,providers=['CPUExecutionProvider'])
            elif name == 'rmn':
                state = torch.load(weights/'rmn_maskfree.pt',map_location='cpu',weights_only=True)
                model = resnet34(weights=None,num_classes=7)
                model.load_state_dict(state,strict=True)
                model = model.eval().to(memory_format=torch.channels_last)
                fuse(model)
            else:
                raise ValueError(f'Unsupported model: {name}')
            self.models[name] = model

    @torch.inference_mode()
    def logits(self, images):
        images = np.asarray(images)
        if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (48,48):
            raise ValueError('输入须为 [N, 48, 48] 的 uint8 灰度人脸')
        output=[]
        flipped=images[:,:,::-1].copy()
        for name, model in self.models.items():
            if name=='tyom':
                views=run(model,images,112,[(1.,False),(1.,True)],'fp32',24,'tyom_resnet18')
                scores=views.mean(1)
            elif name in {'affectlens','affectlens_rehead'}:
                first = model.forward_outputs(images)
                scores = (first.logits.numpy()+model(flipped))/2
                if name == 'affectlens_rehead':
                    scores = scores + first.embedding.numpy() @ self.head_weight + self.head_bias
            elif name=='oxford':
                scores=model(images)
            elif name=='vggm':
                scores=(model(images)+model(flipped))/2
            elif name in {'densenet','cbam'}:
                views=run(model,images,224,[(0.,False),(0.,True)],'int8',16,name)
                scores=views.mean(1)
            elif name=='rmn':
                blocks = []
                for block_start in range(0, len(images), 8):
                    view_scores = []
                    for view in (images[block_start:block_start+8], flipped[block_start:block_start+8]):
                        gray = np.stack([cv2.resize(im,(224,224),interpolation=cv2.INTER_LINEAR) for im in view])
                        pixels = np.repeat(gray[:,None],3,axis=1).astype(np.float32)/255
                        tensor = torch.from_numpy(pixels).contiguous(memory_format=torch.channels_last)
                        view_scores.append(model(tensor).numpy())
                    blocks.append((view_scores[0] + view_scores[1]) / 2)
                scores = np.concatenate(blocks)
            output.append(scores.astype(np.float64))
        values=np.stack(output)
        return values-values.mean(axis=2,keepdims=True)

    def predict_proba(self, images):
        return combine_predictions(self.policy['method'],self.policy['params'],self.logits(images))

    def predict_image(self, path):
        with Image.open(path) as image:
            face=np.asarray(image.convert('L').resize((48,48),Image.Resampling.BILINEAR))
        scores=self.predict_proba(face[None])[0]
        return {'expression':CLASSES_ZH[scores.argmax()],
                'probabilities':dict(zip(CLASSES_ZH,scores.tolist()))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,type=Path)
    parser.add_argument('--weights',default=ROOT/'weights',type=Path)
    parser.add_argument('--policy',default=ROOT/'frozen_policy.json',type=Path)
    parser.add_argument('--threads',default=1,type=int)
    args=parser.parse_args()
    model=CpuEnsemble(args.policy,args.weights,args.threads)
    print(json.dumps(model.predict_image(args.image),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
