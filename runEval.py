from eval_dataset import get_loader as eval_get_loader
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval
import torch
from evaluate import run_eval
import time

def get_eval_loaders():
  eval_loader,vid_count = eval_get_loader(is_train=False,root=args.eval, mv_dir=args.eval_mv,n_work=0,args=args)

  return eval_loader,vid_count

def replace(kval):
  kval = kval.replace("mpconv.1","mpconv")
  kval = kval.replace("conv.0","conv")
  kval = kval.replace("conv.1","batch")
  kval = kval.replace("conv.3","conv1")
  kval = kval.replace("conv.4","batch1")
  #kval = kval.replace("mpconv.1","mpconv")
  return kval


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
      unet_dict.update(pretrain_unet_updated)
      print("Keys not Loaded ",[i for i in unet_dict.keys() if i not in pretrain_unet_updated.keys()])
      net.load_state_dict(unet_dict)


args = parser.parse_args()
print(args)
eval_loader,vid_count = get_eval_loaders()
############### Model ###############
encoder, binarizer, decoder, unet,hypernet = get_models(
  args=args, v_compress=args.v_compress, 
  bits=args.bits,
  encoder_fuse_level=args.encoder_fuse_level,
  decoder_fuse_level=args.decoder_fuse_level,num_vids=vid_count)

nets = [encoder, binarizer, decoder,unet,hypernet]



just_resumed = False
if args.load_model_name:
    print('Loading %s@iter %d' % (args.load_model_name,
                                  args.load_iter))

    resume(args.load_model_name, args.load_iter)
    just_resumed = True



if just_resumed:
  print('Start evaluation...')
  set_eval(nets)

  eval_begin = time.time()
  eval_loss, mssim, psnr = run_eval(nets, eval_loader, args,
      output_suffix='iter%d' % 0)

  print('Evaluation @iter %d done in %d secs' % (
      train_iter, time.time() - eval_begin))
  print('%s Loss   : ' % eval_name
        + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
  print('%s MS-SSIM: ' % eval_name
        + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
  print('%s PSNR   : ' % eval_name
      + '\t'.join(['%.5f' % el for el in psnr.tolist()]))

