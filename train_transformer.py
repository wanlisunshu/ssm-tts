from preprocess import get_dataset, DataLoader, collate_fn_transformer, LJDatasets
from network import *
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from draw_figures import draw_iter_loss_figure, draw_loss_figures
import os
from tqdm import tqdm
import torch as t
import argparse
from pathlib import Path


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validation(m, epoch, device):
    m.eval()
    val_set = LJDatasets(hp.val_path, os.path.join(hp.data_path, 'wavs'))
    val_loader = DataLoader(val_set, batch_size=hp.batch_size, shuffle=False,
                                collate_fn=collate_fn_transformer, num_workers=8)

    val_loss = 0.0
    val_loss1 = 0.0
    val_loss2 = 0.0

    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        pbar.set_description("Validation after epoch %d" % epoch)
        character, mel, mel_input, pos_text, pos_mel, _ = data

        stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)

        character = character.to(device)
        mel = mel.to(device)
        mel_input = mel_input.to(device)
        pos_text = pos_text.to(device)
        pos_mel = pos_mel.to(device)

        loss, loss1, loss2 = m.forward(character, mel, pos_text, pos_mel)

        val_loss += loss.item()
        val_loss1 += loss1.item()
        val_loss2 += loss2.item()
    val_loss = val_loss / len(val_loader.dataset)
    val_loss1 = val_loss1 / len(val_loader.dataset)
    val_loss2 = val_loss2 / len(val_loader.dataset)

    m.train()
    print("Validation average loss in epoch {}: {:9f}, loss1: {:9f}, loss2 {:9f} ".format(epoch, val_loss,
                                                                                                          val_loss1,
                                                                                                          val_loss2))
    return val_loss, val_loss1, val_loss2


def main(output_directory):

    if t.backends.mps.is_available():
        device = t.device("mps")
    else:
        device = t.device('cuda:0')
    print('Using device: ', device)

    # dataset = get_dataset()
    dataset = LJDatasets(hp.train_path, os.path.join(hp.data_path, 'wavs'))
    global_step = 0

    m = Model().to(device)

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).to(device)
    # writer = SummaryWriter()

    loss_epoch_list = []
    loss1_epoch_list = []
    loss2_epoch_list = []
    loss_iter_list = []
    loss1_iter_list = []
    loss2_iter_list = []
    epoch_num_list = []
    iter_num_list = []

    val_loss_epoch_list = []
    val_loss1_epoch_list = []
    val_loss2_epoch_list = []

    epoch_list = []

    loss_iter = 0
    loss1_iter = 0
    loss2_iter = 0
    for epoch in range(hp.epochs):
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=8)
        pbar = tqdm(dataloader)
        loss_epoch = 0
        loss1_epoch = 0
        loss2_epoch = 0
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            character, mel, mel_input, pos_text, pos_mel, _ = data

            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)

            character = character.to(device)
            mel = mel.to(device)
            mel_input = mel_input.to(device)
            pos_text = pos_text.to(device)
            pos_mel = pos_mel.to(device)

            loss, loss1, loss2 = m.forward(character, mel, pos_text, pos_mel)
            loss_iter += loss.item()
            loss_epoch += loss.item()
            loss1_iter += loss1.item()
            loss1_epoch += loss1.item()
            loss2_iter += loss2.item()
            loss2_epoch += loss2.item()            # mel_loss = nn.L1Loss()(mel_pred, mel)
            # post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            # loss = mel_loss + post_mel_loss

            # writer.add_scalars('training_loss',{
            #         'mel_loss':mel_loss,
            #         'post_mel_loss':post_mel_loss,
            #
            #     }, global_step)
            #
            # writer.add_scalars('alphas',{
            #         'encoder_alpha':m.module.encoder.alpha.data,
            #         'decoder_alpha':m.module.decoder.alpha.data,
            #     }, global_step)


            if global_step % hp.image_step == 1:
                loss_iter /= hp.image_step
                loss_iter_list.append(loss_iter)
                loss1_iter /= hp.image_step
                loss1_iter_list.append(loss1_iter)
                loss2_iter /= hp.image_step
                loss2_iter_list.append(loss2_iter)

                iter_num_list.append(global_step)
                loss_iter = 0
                loss1_iter = 0
                loss2_iter = 0

                draw_iter_loss_figure([loss_iter_list, loss1_iter_list, loss2_iter_list], iter_num_list, 'iteration', output_directory)
                # for i, prob in enumerate(attn_probs):
                #
                #     num_h = prob.size(0)
                #     for j in range(4):
                #
                #         x = vutils.make_grid(prob[j*16] * 255)
                #         writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                #
                # for i, prob in enumerate(attns_enc):
                #     num_h = prob.size(0)
                #
                #     for j in range(4):
                #
                #         x = vutils.make_grid(prob[j*16] * 255)
                #         writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
                #
                # for i, prob in enumerate(attns_dec):
                #
                #     num_h = prob.size(0)
                #     for j in range(4):
                #
                #         x = vutils.make_grid(prob[j*16] * 255)
                #         writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)

            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()

            nn.utils.clip_grad_norm_(m.parameters(), 1.)

            # Update weights
            optimizer.step()
            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))
        val_loss, val_loss1, val_loss2 = validation(m, epoch, device)

        val_loss_epoch_list.append(val_loss)
        val_loss1_epoch_list.append(val_loss1)
        val_loss2_epoch_list.append(val_loss2)

        epoch_list.append(epoch)

        loss_epoch /= len(dataloader)
        loss1_epoch /= len(dataloader)
        loss2_epoch /= len(dataloader)
        loss_epoch_list.append(loss_epoch)
        loss1_epoch_list.append(loss1_epoch)
        loss2_epoch_list.append(loss2_epoch)
        epoch_num_list.append(epoch)
        draw_iter_loss_figure([loss_epoch_list, loss1_epoch_list, loss2_epoch_list], epoch_num_list, 'epoch', output_directory)
        draw_loss_figures([loss_epoch_list, loss1_epoch_list, loss2_epoch_list],
                          [val_loss_epoch_list, val_loss1_epoch_list, val_loss2_epoch_list],
                          epoch_list, output_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    args = parser.parse_args()
    Path(args.output_directory).mkdir(parents=True, exist_ok=True)
    main(args.output_directory)
