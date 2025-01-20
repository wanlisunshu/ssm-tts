import torch as t
import argparse
import sys
sys.path.append('.')
from text import text_to_sequence
import numpy as np
# import hyperparams as hp
from network import Model
# import tacotron2.inference
# from contras_regression_loss import ContrastiveRegressionLoss
# from draw_figures import plot_mel
from preprocess import DataLoader, collate_fn_transformer, LJDatasets
# from utils import load_filepaths_and_text
#from tacotron2.inference import generate_mel
from scipy.io.wavfile import write
sys.path.append('./hifi-gan/')
from models import Generator
from env import AttrDict
from inference import load_checkpoint
# from evaluation.generate_energy_distri import init_waveglow
# import toml
import os
from tqdm import tqdm
import json

MAX_WAV_VALUE = 32768.0


def init_hifigan(device):
    config_file = os.path.join('hifi-gan/LJ_V1/', 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    t.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint('hifi-gan/LJ_V1/generator_v1', device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    return generator

def generate_audio(audio_name, mel_outputs, hifigan, hp):
    with t.no_grad():
        y_g_hat = hifigan(mel_outputs)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        # generated_audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    write(audio_name, hp.sr, audio)


# def get_text_and_mel_from_disk(index, text, neg_flag, train_or_val_flag, output_directory):
#     if train_or_val_flag and neg_flag:
#         dataset = LJDatasets(hp.train_path, neg_flag=neg_flag, neg_mel_paths=hp.train_neg_mel_paths)
#         audio_paths_and_text = load_filepaths_and_text(hp.train_path)
#     elif train_or_val_flag and not neg_flag:
#         dataset = LJDatasets(hp.train_path, neg_flag=neg_flag)
#         audio_paths_and_text = load_filepaths_and_text(hp.train_path)
#     elif not train_or_val_flag and neg_flag:
#         dataset = LJDatasets(hp.val_path, neg_flag=neg_flag, neg_mel_paths=hp.val_neg_mel_paths)
#         audio_paths_and_text = load_filepaths_and_text(hp.val_path)
#     elif not train_or_val_flag and not neg_flag:
#         dataset = LJDatasets(hp.val_path, neg_flag=neg_flag)
#         audio_paths_and_text = load_filepaths_and_text(hp.val_path)
#
#     audio_path, text_from_neg = audio_paths_and_text[index][0], audio_paths_and_text[index][1]
#     if not neg_flag:
#         mel = dataset.get_mel(audio_path)
#     else:
#         text_from_neg, mel = dataset.get_neg_mel(audio_path)
#         assert text_from_neg == text, (
#             'text mismatch: text from pretrained model:{}, expected：{}'.format(text_from_neg, text))
#
#     # waveglow = init_waveglow()
#     audio_path = audio_path.strip().split('/')[7]
#     audio_name = output_directory + audio_path + '_generated_by_tacotron.wav'
#     generate_audio(audio_name, mel.unsqueeze(0).cuda(), waveglow)
#     return mel, audio_path
#
#
# def iterative_inference(sequence, output_directory, neg_mel, audio_name):
#     mel_name = 'mel_generated_by_tacotron2'
#     plot_mel(neg_mel.float().data.cpu().numpy(), output_directory, mel_name)
#
#     original_neg_mel = neg_mel.cuda()
#     pos_mel = torch.arange(1, neg_mel.size(1) + 1).unsqueeze(0).cuda()
#     neg_mel = neg_mel.T
#
#     # neg_mel = torch.cat((torch.zeros([1, hp.n_mel_channels]).cuda(), neg_mel[:-1, :]), dim=0)
#     neg_mel = neg_mel.unsqueeze(0)
#     neg_mel = torch.FloatTensor(neg_mel.cpu().data.numpy()).cuda()
#     neg_mel.requires_grad = True
#     pos_text = torch.arange(1, sequence.size(1) + 1).unsqueeze(0).cuda()
#
#     discriminator = Model().cuda()
#     discriminator_path = "result/best_ebm_model_after_epoch_9.pt"
#     discriminator.load_state_dict(torch.load(discriminator_path))
#     # discriminator.train()
#     discriminator.eval()
#     # neg_logits = discriminator.forward(sequence, neg_mel, pos_text, pos_mel)
#
#     criterion = ContrastiveRegressionLoss()
#     optimizer = torch.optim.Adam([neg_mel], lr=0.007)
#     for i in range(100):
#         optimizer.zero_grad()
#         fake_logits = discriminator.forward(sequence, neg_mel, pos_text, pos_mel)
#         # sigmoid = torch.nn.Sigmoid()
#         # fake_logits = sigmoid(fake_logits)
#         print("iteration: {} energy score: {}".format(i, fake_logits.item()))
#         if i is not 0:
#             mel_name = 'mel_after_iteration_{}'.format(i)
#             plot_mel(neg_mel.squeeze(0).T.float().data.cpu().numpy(), output_directory, mel_name)
#             bias_mel_name = 'difference_mel_after_iteration_{}'.format(i)
#             plot_mel((neg_mel.squeeze(0).T - original_neg_mel).float().data.cpu().numpy(), output_directory, bias_mel_name)
#
#             waveglow = init_waveglow()
#             audio_path = (output_directory + audio_name + '_updated_after_epoch_{}.wav').format(i)
#             generate_audio(audio_path, neg_mel.permute(0, 2, 1), waveglow)
#
#         fake_logits.backward()
#         optimizer.step()


def iterative_inference_batch(t2_mel, pos_mel,audio_name, hifigan, output_directory,
                              discriminator_path, file_list, step, device):
    discriminator = Model().to(device)
    discriminator.load_state_dict(t.load(discriminator_path, map_location=device)['model'])
    discriminator.eval()

    learning_rate = 0.01
    optimizer = t.optim.SGD([t2_mel], lr=learning_rate)
    # iter_num = 100
    for i in range(step):
        optimizer.zero_grad()
        _, _, _, score = discriminator.forward(t2_mel, pos_mel)
        # sigmoid = torch.nn.Sigmoid()
        # fake_logits = sigmoid(fake_logits)
        # if i is 0:
            # print("iteration: {} energy score: {}".format(i, score.item()))
            # energy_before.append(score.item())
        # if i is not 0:
            # mel_name = 'mel_after_iteration_{}'.format(i)
            # plot_mel(neg_mel.squeeze(0).T.float().data.cpu().numpy(), output_directory, mel_name)
            # bias_mel_name = 'difference_mel_after_iteration_{}'.format(i)
            # plot_mel((neg_mel.squeeze(0).T - original_neg_mel).float().data.cpu().numpy(), output_directory, bias_mel_name)
        score = t.mean(t.mean(t.mean(score, 0), 0))
        score.backward()
        optimizer.step()

    relative_path = discriminator_path.split('/')[1].split('.')[0]
    # hifigan, vocoder_train_setup, denoiser = t.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
    # hifigan.to(device)
    audio_path = output_directory + '_'+ relative_path + '_step_' + str(step)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)


    audio_path += '/' + audio_name.split('.')[0] + '_updated.wav'
    generate_audio(audio_path, t2_mel.permute(0, 2, 1), hifigan, hp)

    # print("iteration: {} energy score: {}".format(i, fake_logits.item()))
    # energy_after.append(fake_logits.item())
    file_list.append(audio_name.split('.')[0])
    temp = {}
    temp['file_list'] = file_list
    # temp['energy_before'] = energy_before
    # temp['energy_after'] = energy_after
    # total = sum(list(map(lambda x: x[0] > 0 and x[1] < 0, zip(energy_before, energy_after))))
    # print("transforming_rate: ", total / len(energy_before))
    # temp['transforming_rate'] = total / len(energy_before)
    stats_data = output_directory + relative_path + '_step_' + str(step) + '/stats_data.pt'
    t.save(temp, stats_data)
    return file_list


def iterative_inference_multi(output_directory, discriminator_path,  step):
    if t.backends.mps.is_available():
        device = t.device("mps")
    else:
        device = t.device('cuda:0')
    print('Using device: ', device)

    dataset = LJDatasets(hp.val_path, os.path.join(hp.data_path, 'wavs'))
    # audio_paths_and_text = load_filepaths_and_text(hp.val_path)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn_transformer, num_workers=2)
    # energy_before = []
    # energy_after = []
    file_list = []
    pbar = tqdm(val_loader)
    hifigan = init_hifigan(device)
    for i, data in enumerate(pbar):
        # pbar.set_description("Validation after epoch %d" % epoch)
        character, ref_mel, t2_mel, pos_text, pos_mel, text_len, audio_name = data

        character = character.to(device)
        ref_mel = ref_mel.to(device)
        t2_mel = t2_mel.to(device)
        pos_text = pos_text.to(device)
        pos_mel = pos_mel.to(device)
    # for batch in iter(audio_paths_and_text):
    #     audio_path, text_from_neg = batch[0], batch[1]
    #     text_from_neg, mel = dataset.get_neg_mel(audio_path)
    #     audio_path = audio_path.strip().split('/')[7]
    #     sequence = np.array(text_to_sequence(text_from_neg, hp['text_cleaners']))[None, :]
    #     sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda()

        # waveglow = init_waveglow()
        # audio_path = output_directory + 'tacotron_audio/' + audio_path.split('.')[0] + '_generated.wav'
        # generate_audio(audio_path, mel.unsqueeze(0).cuda() , waveglow)

        file_list = iterative_inference_batch(t2_mel, pos_mel, audio_name[0], hifigan, output_directory,
                                              discriminator_path, file_list, step, device)


# def iterative_inference_single_on_disk(index, text, output_directory, neg_flag, train_or_val_flag):
#     sequence = np.array(text_to_sequence(text, hp.text_cleaners))[None, :]
#     sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda()
#
#     mel, audio_path = get_text_and_mel_from_disk(index, text, neg_flag, train_or_val_flag, output_directory)
#     iterative_inference(sequence, output_directory, mel, audio_path)


# def iterative_inference_online(text, output_directory):
#     sequence = np.array(text_to_sequence(text, hp.text_cleaners))[None, :]
#     sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda()
#
#     mel = generate_mel(sequence)
#     iterative_inference(sequence, output_directory, mel)


if __name__ == '__main__':
    # iterative inference
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--hp', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('-s', '--step', type=int,
                        required=False, help='num of steps for iterative inference')
    parser.add_argument('-m', '--model_path', type=str,
                        required=False, help='path of best model ')
    args = parser.parse_args()
    if args.hp:
        import args.hp as hp
    else:
        import hyperparams as hp

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    # waveglow = init_waveglow()

    # text = "thus disposing of the objection so long raised against the industrial employment of prisoners in Newgate."
    # # iterative_inference_online(text, args.output_directory)
    # iterative_inference_single_on_disk(1379, text, args.output_directory, True, False)
    # discriminator_path_list = ["result/best_ebm_model_after_epoch_291.pt"]

    # for path in discriminator_path_list:
    iterative_inference_multi(args.output_directory, args.model_path, args.step)
