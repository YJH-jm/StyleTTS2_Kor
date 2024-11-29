from glob import glob

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import time
import yaml
from munch import Munch
import numpy as np
from scipy.io import wavfile
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner

from pathlib import Path


import phonemizer





from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert



class StyleTTS2:
    def __init__(self, model_path = "Models/LJSpeech/epoch_2nd_00100.pth", config_path="Models/LJSpeech/config.yml"):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

        self.textcleaner = TextCleaner()
        
        if os.path.isfile(config_path) and os.path.isfile(model_path):
            self.model = self.load_model(model_path, config_path)
        
        
        
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4


    def load_model(self, model_path=None, config_path=None):
        

        self.config = yaml.safe_load(open(config_path))

        # load pretrained ASR model
        ASR_config = self.config.get('ASR_config', False)
        ASR_path = self.config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model -> 학습 후 수정 필요
        BERT_path = self.config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)


        self.model_params = recursive_munch(self.config['model_params'])
        model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        params_whole = torch.load(model_path, map_location='cpu')
        params = params_whole['net']

        for key in model:
            if key in params:
                print('%s loaded' % key)
                try:
                    model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model[key].load_state_dict(new_state_dict, strict=False)
       
        _ = [model[key].eval() for key in model]

        return model

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)    
    

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def inference(self,
                  text: str,
                  ref_audio_path=None,
                  alpha: float =0.3,
                  beta: float =0.7,
                  diffusion_steps:int=5,
                  embedding_scale=1):
        
        ref_s = None
        if ref_audio_path is not None:
            ref_s = self.compute_style(ref_audio_path)

        noise = torch.randn(1,1,256).to(self.device)

        # text preprocessing 
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text]) # text의 발음 기호 생성 
        ps = word_tokenize(ps[0]) # len(ps) = 25, 발음 기호 문자열을 단어 단위로 나늠 
        ps = ' '.join(ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0) # start token 
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)  # int64, torch.Size([1, 207])

        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device) 
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask) # torch.Size([207])
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int()) # torch.Size([1, 207, 768])
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) # torch.Size([1, 512, 207])


            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            # print(ref_s)
            if ref_s is not None:
                ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
                s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                        s, input_lengths, text_mask)
            # d-> torch.Size([1, 207, 640]) , duration encoder output
        
            x, _ = self.model.predictor.lstm(d) # x -> torch.Size([1, 207, 512])
            duration = self.model.predictor.duration_proj(x) # torch.Size([1, 207])


            duration = torch.sigmoid(duration).sum(axis=-1) # torch.Size([1, 207])
            pred_dur = torch.round(duration.squeeze()).clamp(min=1) ## torch.Size([207])

            if ref_s is None:
                pred_dur[-1] += 5 # add for single speark 

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data)) # # torch.Size([207, 483])

            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)


     
            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            print(self.model_params.decoder.type)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)


            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        
        # return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        out = out.squeeze().cpu().numpy()
        return out if ref_s is None else out[..., : -50]


        



if __name__ == "__main__":

    text = '''Taking care of your mental health is a journey, and there\'s no one-size-fits-all approach.'''

    saved_name = "11.wav"

    # for single 
    # model_path = "Models/LJSpeech/epoch_2nd_00100.pth", 
    # config_path="Models/LJSpeech/config.yml"
    # styletts2 = StyleTTS2()
    # start = time.time()
    # wav = styletts2.inference(text)
    # rtf = (time.time() - start) / (len(wav) / 24000)
    # print(f"RTF = {rtf:5f}")

    # wavfile.write(saved_name, 24000, wav)


    # for multi and clone
    model_path = "Models/LibriTTS/epochs_2nd_00020.pth"
    config_path = "Models/LibriTTS/config.yml"
    styletts2 = StyleTTS2(model_path=model_path, config_path=config_path)
    reference_dicts = {}
    # reference_dicts['696_92939'] = "Demo/reference_audio/696_92939_000016_000006.wav"
    # reference_dicts['1789_142896'] = "Demo/reference_audio/1789_142896_000022_000005.wav"
    reference_dicts['3'] = "Demo/reference_audio/3.wav"

    TEXT_ROOT = "/workspace/ai/jhyoo/nlp/Step1/eng"

    if not os.path.isdir(TEXT_ROOT):
        print(f"경로 {TEXT_ROOT} 존재하지 않음")
        import
        sys.exit()
    pkl_list = glob()

    for k, path in reference_dicts.items():
        start = time.time()
        
        wav = styletts2.inference(text, ref_audio_path=path)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")

        # wavfile.write(f'output_clone_{k}.wav', 24000, wav)
        wavfile.write(saved_name, 24000, wav)

