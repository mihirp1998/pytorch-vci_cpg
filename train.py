import numpy as np
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataset import get_loader
from eval_dataset import get_loader as eval_get_loader
from evaluate import run_eval
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval
from util import prepare_inputs, forward_ctx

args = parser.parse_args()
print(args)

############### Data ###############
train_loader = get_loader(
  is_train=True,
  root=args.train, mv_dir=args.train_mv,n_work=2,
  args=args
)

def get_eval_loaders():
  # We can extend this dict to evaluate on multiple datasets.
  # change this later
  eval_loader,vid_count = eval_get_loader(is_train=False,root=args.eval, mv_dir=args.eval_mv,n_work=0,args=args)
 
  return eval_loader,vid_count


writer = SummaryWriter()

############### Model ###############
encoder, binarizer, decoder, unet,hypernet = get_models(
  args=args, v_compress=args.v_compress, 
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level,num_vids=train_loader.vid_count)

nets = [encoder, binarizer, decoder,unet,hypernet]
#if unet is not None:
#  nets.append(unet)

# print(nets[0].rnn2.weights.shape,"rnn2")
gpus = [int(gpu) for gpu in args.gpus.split(',')]
if len(gpus) > 1:
  print("Using GPUs {}.".format(gpus))
  for net in nets:
    net = nn.DataParallel(net, device_ids=gpus)

params = [{'params': net.parameters()} for net in nets]

solver = optim.Adam(
    params,
    lr=args.lr)

milestones = [int(s) for s in args.schedule.split(',')]
scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)

if not os.path.exists(args.model_dir):
  print("Creating directory %s." % args.model_dir)
  os.makedirs(args.model_dir)

def replace(kval):
  kval = kval.replace("mpconv.1","mpconv")
  kval = kval.replace("conv.0","conv")
  kval = kval.replace("conv.1","batch")
  kval = kval.replace("conv.3","conv1")
  kval = kval.replace("conv.4","batch1")
  #kval = kval.replace("mpconv.1","mpconv")
  return kval

############### Checkpoints ###############


def resume(load_name, index):
  names = ['encoder', 'binarizer', 'decoder', 'unet']
  for net_idx, net in enumerate(nets[:4]):
    if net_idx != 3:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_epoch_{:08d}.pth'.format(args.model_dir, load_name, name, index)
      print('Loading %s from %s...' % (name, checkpoint_path))
      net.load_state_dict(torch.load(checkpoint_path))
    else:
      name = names[net_idx]
      unet_dict = net.state_dict()
      checkpoint_path = '{}/{}_{}_epoch_{:08d}.pth'.format(args.model_dir, load_name, name, index)
      print('Loading %s from %s...' % (name, checkpoint_path))
      pretrain_unet = torch.load(checkpoint_path)
      # replace key names
      pretrain_unet = {replace(k): v for k, v in pretrain_unet.items()}
      # print("pretrain unet",pretrain_unet.keys())
      pretrain_unet_updated = {}
      
      for k, v in pretrain_unet.items():
        if k in unet_dict:
          pretrain_unet_updated[k] = v
        else:
          print("WARNING: Unable to Load Params {}".format(k))

      pretrain_unet_2 = torch.load("vcii_model_params/demo_unet_00010000.pth")
      for k, v in pretrain_unet_2.items():
        if k in unet_dict:
          pretrain_unet_updated[k] = v
        else:
          print("WARNING: Unable to Load Params {}".format(k))

      unet_dict.update(pretrain_unet_updated)
      print("Keys not Loaded ",[i for i in unet_dict.keys() if i not in pretrain_unet_updated.keys()])
      net.load_state_dict(unet_dict)
  hypernet = nets[4]
  hypernet.load_state_dict(torch.load("vcii_model_params/demo_hypernet_00010000.pth"))



def save(index):

  # for net_idx, net in enumerate(nets):
  #   if net is not None:
  torch.save(unet.state_dict(),'{}/{}_{}_{:08d}.pth'.format(args.model_dir, args.save_model_name,"unet", index))
  torch.save(hypernet.state_dict(),'{}/{}_{}_{:08d}.pth'.format(args.model_dir, args.save_model_name,"hypernet", index))


############### Training ###############

train_iter = 0
just_resumed = False
if args.load_model_name:
    print('Loading %s@iter %d' % (args.load_model_name,
                                  args.load_iter))

    resume(args.load_model_name, args.load_iter)
    train_iter = args.load_iter
    scheduler.last_epoch = train_iter - 1
    just_resumed = True

all_losses = []
solver.zero_grad()
loss_mini_batch =0
while True:
#     print("while true began")
    for batch, (crops, ctx_frames, _ ,id_num) in enumerate(train_loader):
        # print("batch not starting")
        scheduler.step()
        train_iter += 1
        # print(crops.shape,"shape")
        #crops = pickle.load(open("crop1.p","rb"))
        #ctx_frames = pickle.load(open("ctx_frames1.p","rb"))
        # id_num = pickle.load(open("id_num.p","rb"))
        if train_iter > args.max_train_iters:
          break

        batch_t0 = time.time()

        # solver.zero_grad()

        id_num = Variable(torch.tensor(id_num).cuda())

        # Init LSTM states.
        (encoder_h_1, encoder_h_2, encoder_h_3,
         decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm(
            batch_size=(crops[0].size(0) * args.num_crops), height=crops[0].size(2),
            width=crops[0].size(3), args=args)

        wenc,wdec,wbin,unet_kernels,unet_bias = hypernet(id_num)
        # Forward U-net.
        if args.v_compress:
            unet_output1, unet_output2 = forward_ctx(unet, ctx_frames,unet_kernels,unet_bias)
        else:
            unet_output1 = Variable(torch.zeros(args.batch_size,)).cuda()
            unet_output2 = Variable(torch.zeros(args.batch_size,)).cuda()

        res, frame1, frame2, warped_unet_output1, warped_unet_output2 = prepare_inputs(
            crops, args, unet_output1, unet_output2)

        losses = []

        bp_t0 = time.time()
        _, _, height, width = res.size()

        out_img = torch.zeros(1, 3, height, width).cuda() + 0.5

        for _ in range(args.iterations):
            if args.v_compress and args.stack:
                encoder_input = torch.cat([frame1, res, frame2], dim=1)
            else:
                encoder_input = res

            # Encode.
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
                warped_unet_output1, warped_unet_output2,wenc)

            # Binarize.
            codes = binarizer(encoded,wbin)

            # Decode.
            (output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
                warped_unet_output1, warped_unet_output2,wdec)

            res = res - output
            out_img = out_img + output.data
            losses.append(res.abs().mean())

        bp_t1 = time.time()
        all_losses.append(losses)

        loss = sum(losses) / args.iterations
        loss = loss/args.update
        loss.backward()

        loss_mini_batch += loss.data[0]

        for net in [encoder, binarizer, decoder, unet,hypernet]:
            if net is not None:
                torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)

        # solver.step()

        batch_t1 = time.time()


        if train_iter % 100 == 0:
            print('Loss at each step:')
            print(('{:.4f} ' * args.iterations +
                   '\n').format(* [l.data[0] for l in losses]))

        if train_iter % args.checkpoint_iters == 0:
            save(train_iter)

        if train_iter % args.update==0:
            solver.step()
            solver.zero_grad()
            print('[TRAIN] Iter[{}]; LR: {}; Loss: {:.6f}; Backprop: {:.4f} sec; Batch: {:.4f} sec'.format(train_iter, scheduler.get_lr()[0], loss_mini_batch, bp_t1 - bp_t0, batch_t1 - batch_t0))
            loss_mini_batch =0
            print(('{:.4f} ' * args.iterations +'\n').format(* [l.data[0] for l in np.array(losses)]))
            all_losses = []

        if just_resumed or train_iter % args.eval_iters == 0:
            print('Start evaluation...')

            set_eval(nets)

            eval_loaders,vid_count = get_eval_loaders()

            eval_begin = time.time()
            eval_loss, mssim, psnr = run_eval(nets, eval_loaders, args,
                output_suffix='iter%d' % train_iter)

            writer.add_scalar('data/mean_1st iter',eval_loss.tolist()[0], train_iter)
            writer.add_scalar('data/mean_05th iter',eval_loss.tolist()[5], train_iter)
            writer.add_scalar('data/mean_10th iter',eval_loss.tolist()[9], train_iter)
            writer.add_scalar('data/mean_sum_iter',sum(eval_loss.tolist()), train_iter)
            print('Evaluation @iter %d done in %d secs' % (
                train_iter, time.time() - eval_begin))
            print('%s Loss   : ' % '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
            print('%s MS-SSIM: ' % '\t'.join(['%.5f' % el for el in mssim.tolist()]))
            print('%s PSNR   : ' % '\t'.join(['%.5f' % el for el in psnr.tolist()]))

            set_train(nets)
            just_resumed = False


    if train_iter > args.max_train_iters:
      print('Training done.')
      break
