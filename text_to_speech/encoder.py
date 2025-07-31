from pydub import AudioSegment
# from resemblyzer import VoiceEncoder
import numpy as np
from TTS.api import TTS
import torch
from LSTM import *
from audio import *
from hyperparams import *
import os

print("当前工作目录：", os.getcwd())

text="他叫XXX……是我这辈子，最想一直缠着的人。我爱他，不需要理由，也不需要故事。只要他在……我就刚刚好。"


def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples

samples1 = preprocess_audio("qingyunzhi.m4a")
samples2 = preprocess_audio("chuntingxue.m4a")

encoder = VoiceEncoder()
embed1 = encoder.embed_utterance(samples1)
embed2 = encoder.embed_utterance(samples2)

sim = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

print(f"Cosine similairty: {sim:.4f}")

# save
# np.save('qingyunzhi_embed.npy', embed1)
# np.save('chuntingxue_embed.npy', embed2)


device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

def convert_m4a_to_wav(path_in, path_out):
    audio = AudioSegment.from_file(path_in, format="m4a")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(path_out, format="wav")


def convert_mp3_to_wav(input_mp3, output_wav):
    audio = AudioSegment.from_mp3(input_mp3)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)

    audio.export(output_wav, format="wav")
    print(f"转换完成: {input_mp3} → {output_wav}")


convert_mp3_to_wav("donggege.mp3", "donggege.wav")

convert_m4a_to_wav("qingyunzhi.m4a", "qingyunzhi.wav")
convert_m4a_to_wav("chuntingxue.m4a", "chuntingxue.wav")

tts.tts_to_file(
    text=text,
    speaker_wav="donggege.wav",
    file_path="output_qingyunzhi.wav",
    language="zh-cn"
)


print("Done!")