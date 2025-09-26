import torch
import librosa
import numpy as np
import torch.nn.functional as F
from rmvpe.inference import RMVPE

def extract_rmvpe_embedding(audio_file, model_path="pretrain/rmvpe.pt"):
    # 加载模型
    rmvpe = RMVPE(model_path)
    
    # 加载音频
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(rmvpe.dtype).to(rmvpe.device)
    
    # 提取mel谱
    mel_extractor = rmvpe.mel_extractor.to(rmvpe.device)
    mel = mel_extractor(audio_tensor, center=True).to(rmvpe.dtype)
    
    # 获取intermediate特征
    with torch.no_grad():
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='reflect')
        
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = rmvpe.model.cnn(rmvpe.model.unet(mel)).transpose(1, 2).flatten(-2)
        x = x.contiguous()
        
        embedding = torch.mean(x, dim=1)
        return embedding.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    embedding = extract_rmvpe_embedding("test.wav")
    print(f"RMVPE embedding shape: {embedding.shape}")
    np.save("rmvpe_embedding.npy", embedding)
