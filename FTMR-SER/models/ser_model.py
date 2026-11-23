import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class Ser_Model(nn.Module):
    def __init__(self):
        super(Ser_Model, self).__init__()
        
        # --- Constants ---
        embed_dim = 128
        num_classes = 6
        

        try:
            from models.ser_spec import SER_AlexNet
            self.alexnet_model = SER_AlexNet(num_classes=num_classes, in_ch=3, pretrained=True)
        except ImportError:

            print("Warning: SER_AlexNet class not found. Using a dummy CNN.")
            self.alexnet_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16 * 128 * 128, 9216)
            )

        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(9216, embed_dim)
        

        self.lstm_mfcc = nn.LSTM(
            input_size=40, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.5, 
            bidirectional=True
        )
        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(153600, embed_dim)
        

        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(embed_dim * 2, 149)
                            

        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2‑base",
            local_files_only=True
        )
      
        # for param in self.wav2vec2_model.parameters():
        #     param.requires_grad = False

        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, embed_dim)
        

        self.gate_spec = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.gate_mfcc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.gate_wav = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        

        contrastive_dim = 64
        self.contrastive_proj_spec_mfcc = nn.Sequential(
            nn.Linear(embed_dim * 2, contrastive_dim),
            nn.ReLU(),
            nn.Linear(contrastive_dim, contrastive_dim)
        )
        self.contrastive_proj_wav = nn.Sequential(
            nn.Linear(768, contrastive_dim),  
            nn.ReLU(),
            nn.Linear(contrastive_dim, contrastive_dim)
        )
        

        self.student_head_spec = nn.Linear(embed_dim, num_classes)
        self.student_head_mfcc = nn.Linear(embed_dim, num_classes)
        self.student_head_wav = nn.Linear(embed_dim, num_classes) 
        

        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(embed_dim * 3, embed_dim)
        self.post_att_layer_2 = nn.Linear(embed_dim, embed_dim)
        self.post_att_layer_3 = nn.Linear(embed_dim, num_classes)

    def forward(self, audio_spec, audio_mfcc, audio_wav):    

        device = audio_spec.device
        audio_mfcc = audio_mfcc.to(device)
        audio_wav = audio_wav.to(device)
        

        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        

        cnn_output = self.alexnet_model(audio_spec)
        if isinstance(cnn_output, tuple):
            audio_spec_cnn, _ = cnn_output
        else:
            audio_spec_cnn = cnn_output
        audio_spec_flat = torch.flatten(audio_spec_cnn, 1)
        audio_spec_d = self.post_spec_dropout(audio_spec_flat)
        E_spec = F.relu(self.post_spec_layer(audio_spec_d), inplace=False)
        

        audio_mfcc_lstm, _ = self.lstm_mfcc(audio_mfcc)
        audio_mfcc_flat = torch.flatten(audio_mfcc_lstm, 1)
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_flat)
        E_mfcc = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False)
        

        spec_mfcc = torch.cat([E_spec, E_mfcc], dim=-1)

   
        f_freq = self.contrastive_proj_spec_mfcc(spec_mfcc)
        

        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d), inplace=False)
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)
        

        wav_features_seq = self.wav2vec2_model(audio_wav).last_hidden_state
        audio_wav_pooled = torch.matmul(audio_spec_mfcc_att_p, wav_features_seq)
        audio_wav_unpooled_d = audio_wav_pooled.reshape(audio_wav_pooled.shape[0], -1)
        

        f_time = self.contrastive_proj_wav(audio_wav_unpooled_d)
        

        audio_wav_d = self.post_wav_dropout(audio_wav_unpooled_d)
        E_wav = F.relu(self.post_wav_layer(audio_wav_d), inplace=False)
        

        gate_spec = self.gate_spec(E_spec)
        gate_mfcc = self.gate_mfcc(E_mfcc)
        gate_wav = self.gate_wav(E_wav)

        E_spec = E_spec * gate_spec
        E_mfcc = E_mfcc * gate_mfcc
        E_wav = E_wav * gate_wav


        S_spec = self.student_head_spec(E_spec)
        S_mfcc = self.student_head_mfcc(E_mfcc)
        S_wav = self.student_head_wav(E_wav)
        

        audio_att = torch.cat([E_spec, E_mfcc, E_wav], dim=-1)  
        
        audio_att_d_1 = self.post_att_dropout(audio_att)
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)
        audio_att_d_2 = self.post_att_dropout(audio_att_1)
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)
        M_logits = self.post_att_layer_3(audio_att_2)
        

        output = {

            'M': M_logits,

            'S_spec': S_spec,
            'S_mfcc': S_mfcc,
            'S_wav': S_wav,

            'F_freq': f_freq,
            'F_time': f_time,

            'F1': E_wav,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': M_logits
        }          
        
        return output
