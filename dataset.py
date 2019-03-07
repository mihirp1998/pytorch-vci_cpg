import os
import os.path
import glob

import torch
import torch.utils.data as data
import numpy as np
import random
import cv2
from collections import defaultdict
import pickle
from joblib import delayed
from joblib import Parallel
random.seed(42)

def get_loader(is_train, root, mv_dir,n_work, args):
    print('\nCreating loader for %s...' % root)

    dset = ImageFolder(
        is_train=is_train,
        root=root,
        mv_dir=mv_dir,
        args=args,
    )

    loader = data.DataLoader(
        dataset=dset,
        batch_size=1,
        shuffle=is_train,
        num_workers=0
    )

    # print('Loader for {} images ({} batches) created.'.format(
    #     len(dset), len(loader))
    # )

    return dset


def default_loader(path):
    cv2_img = cv2.imread(path)
    if cv2_img.shape is None:
        print(path)
        print(cv2_img)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    width, height, _ = cv2_img.shape
    if width % 16 != 0 or height % 16 != 0:
        cv2_img = cv2_img[:(width//16)*16, :(height//16)*16]

    return cv2_img


def read_bmv(fn):
    a = cv2.imread(fn, 0)
    if a is not None:
        width, height = a.shape
        if width % 16 != 0 or height % 16 != 0:
            a = a[:(width//16)*16, :(height//16)*16]

        return a[:, :, np.newaxis].astype(float) - 128.0
    else:
        print('no bmv found (it\'s okay if not too often)', fn)
        return None


def get_bmv(img, fns):
    before_x, before_y, after_x, after_y = fns

    bmvs = [read_bmv(before_x),
            read_bmv(before_y),
            read_bmv(after_x),
            read_bmv(after_y)]

    if bmvs[0] is None or bmvs[1] is None:
        if 'ultra_video_group' in before_x:
            # We need HW to be (16n1, 16n2).
            bmvs[0] = np.zeros((1072, 1920, 1))
            bmvs[1] = np.zeros((1072, 1920, 1))
        else:
            bmvs[0] = np.zeros((288, 352, 1))
            bmvs[1] = np.zeros((288, 352, 1))
    else:
        bmvs[0] = bmvs[0] * (-2.0)
        bmvs[1] = bmvs[1] * (-2.0)        
 
    if bmvs[2] is None or bmvs[3] is None:
        if 'ultra_video_group' in before_x:
            bmvs[2] = np.zeros((1072, 1920, 1))
            bmvs[3] = np.zeros((1072, 1920, 1))
        else:
            bmvs[2] = np.zeros((288, 352, 1))
            bmvs[3] = np.zeros((288, 352, 1))
    else:
        bmvs[2] = bmvs[2] * (-2.0)
        bmvs[3] = bmvs[3] * (-2.0)        

    return bmvs


def crop_cv2(img, patch):
    height, width, c = img.shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    start_x =0
    start_y =0
    return img[start_x : start_x + height, start_y : start_y + width]


def flip_cv2(img, patch):
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()

        assert img.shape[2] == 13, img.shape
        # height first, and then width. but BMV is (width, height)... sorry..
        img[:, :, 9] = img[:, :, 9] * (-1.0)
        img[:, :, 11] = img[:, :, 11] * (-1.0)
    return img


# (Close, far)
def get_group_filenames(filename, img_idx, distance1, distance2):
    dtype = filename[-3:]
    assert filename[-4] == '.'
    code = filename[:-4].split('_')[-1]

    # I 2 3 D 5 6 D 8 9 B 11 12 I
    if img_idx % 12 in [3, 6, 9, 0]:
        delta_close = distance1
        delta_far = distance2 * (-1)
    else:
        delta_close = distance1 * (-1)
        delta_far = distance2

    filenames = [filename[:-4 - len(code)] + str(img_idx + delta_close).zfill(len(code)) + '.%s' % dtype,
                 filename[:-4 - len(code)] + str(img_idx).zfill(len(code)) + '.%s' % dtype,
                 filename[:-4 - len(code)] + str(img_idx + delta_far).zfill(len(code)) + '.%s' % dtype]

    return filenames


def get_bmv_filenames(mv_dir, main_fn):

    fn = main_fn.split('/')[-1][:-4]

    return (os.path.join(mv_dir, fn + '_before_flow_x_0001.jpg'),
            os.path.join(mv_dir, fn + '_before_flow_y_0001.jpg'),
            os.path.join(mv_dir, fn + '_after_flow_x_0001.jpg'),
            os.path.join(mv_dir, fn + '_after_flow_y_0001.jpg'))


def get_identity_grid(shape):
    width, height = shape
    grid = np.zeros((width, height, 2))
    for i in range(width):
        for j in range(height):
            grid[i, j, 0] = float(j) * (2.0 / (height - 1.0)) - 1.0
            grid[i, j, 1] = float(i) * (2.0 / (width - 1.0)) - 1.0
    return grid


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1) #w, h, 9
    img = np.swapaxes(img, 0, 2) #9, h, w
    return torch.from_numpy(img).float()


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, is_train, root, mv_dir, args):

        self.is_train = is_train
        self.root = root
        self.args = args
        self.mv_dir = mv_dir

        self.patch = args.patch
        self.loader = default_loader
        self.v_compress = args.v_compress
        self._num_crops = args.num_crops

        self.identity_grid = None

        self._load_image_list()
        self.perVideoInfo()
        #self.vid_freq,self.vid2id = self.genIds()
        pickle.dump(self.vid2id,open("train_dict100.p","wb"))
        #self.vid2id = pickle.load(open("train_dict.p","rb"))
        if is_train:
            random.shuffle(self.imgs)

        print('\tdistance=%d/%d' % (args.distance1, args.distance2))

    def perVideoInfo(self):
        self.d = defaultdict(lambda: [])
        self.id_names = []
        vid = 0 
        for i in self.imgs:
            id_val = i.split("/")[2][:-9]
            self.d[id_val].append(i)
            self.id_names.append(id_val)
        self.id_names = list(set(self.id_names))
        
        self.vid2id = dict()
        for w, c in self.d.items():
            self.vid2id[w] = vid
            vid +=1
        self.vid_count = len(self.vid2id.keys())
        print("num of videos {} ".format(self.vid_count))


    def _load_image_list(self):
        self.imgs = []
        dist1, dist2 = self.args.distance1, self.args.distance2

        if self.v_compress:
            if dist1 == 6 and dist2 == 6:
                positions = [7]
            elif dist1 == 3 and dist2 == 3:
                positions = [4, 10]
            elif dist1 == 1 and dist2 == 2: 
                positions = [2, 3, 5, 6, 8, 9, 11, 0]
            else:
                assert False, 'not implemented.'

        for filename in glob.iglob(self.root + '/*png'):
            img_idx = int(filename[:-4].split('_')[-1])

            if self.args.v_compress:
                if not (img_idx % 12 in positions):
                    continue
                if all(os.path.isfile(fn) for fn in
                       get_group_filenames(
                            filename, img_idx, dist1, dist2)):

                    self.imgs.append(filename)
            else:
                if (img_idx % 12) != 1:
                    continue
                if os.path.isfile(filename):
                    self.imgs.append(filename)

        print('%d images loaded.' % len(self.imgs))

    def get_group_data(self, filename):
        img_idx = int(filename[:-4].split('_')[-1])

        filenames = get_group_filenames(
            filename, img_idx, 
            self.args.distance1,
            self.args.distance2)
        assert all(os.path.isfile(fn) for fn in filenames), filenames
        assert len(filenames) == 3

        imgs_ = [self.loader(fn) for fn in filenames]

        main_fn = filenames[1]
        return np.concatenate(imgs_, axis=2), main_fn

    def get_frame_data(self, filename):
        img = self.loader(filename)
        return img, filename

    def load_data(self,filename):
        if self.v_compress:
            img, main_fn = self.get_group_data(filename)
        else:
            img, main_fn = self.get_frame_data(filename)

        if self.args.warp:
            bmv = np.concatenate(get_bmv(
                img, get_bmv_filenames(self.mv_dir, main_fn)), axis=2)

            img_idx = int(main_fn[:-4].split('_')[-1])

            assert bmv.shape[2] == 4
            # I 2 3 D 5 6 D 8 9 B 11 12 I
            if img_idx % 12 in [3, 6, 9, 0]:
                # From (before, after) to (close, far).
                tmp = bmv[:, :, :2].copy()
                bmv[:, :, :2] = bmv[:, :, 2:4].copy()
                bmv[:, :, 2:4] = tmp
            else:
                if self.args.distance1 == 1:
                    assert img_idx % 12 in [2, 5, 8, 11]

            width, height, c = img.shape
            # For both train and eval. We use full context.
            bmv[:, :, 0] = bmv[:, :, 0] / height
            bmv[:, :, 1] = bmv[:, :, 1] / width
            bmv[:, :, 2] = bmv[:, :, 2] / height
            bmv[:, :, 3] = bmv[:, :, 3] / width

        img = np.concatenate([img, bmv], axis=2)

        assert img.shape[2] == 13
        if self.is_train:
            # If use_bmv, * -1.0 on bmv for flipped images.
            img = flip_cv2(img, self.patch)

        if self.identity_grid is None:
            self.identity_grid = get_identity_grid(img.shape[:2])

        img[..., 9 :11] += self.identity_grid
        img[..., 11:13] += self.identity_grid

        # Split img.
        ctx_frames = img[..., [0, 1, 2, 6, 7, 8]]

        assert img.shape[2] == 13
        assert ctx_frames.shape[2] == 6


        # CV2 cropping in CPU is faster.
        if self.is_train:
            crops = []
            for i in range(self._num_crops):
                # crop = crop_cv2(img, self.patch)
                crop = img
                crop[..., :9] /= 255.0
                crops.append(np_to_torch(crop))
            data = crops
        else:
            img[..., :9] /= 255.0
            data = np_to_torch(img)

        ctx_frames /= 255.0
        ctx_frames = np_to_torch(ctx_frames)

        self.data_s.append(torch.stack(data))
        self.ctx_frames_s.append(ctx_frames)
        self.main_fn_s.append(main_fn)


    def __getitem__(self, index):
        random.shuffle(self.id_names)
        vidname = self.id_names[index]
        print("vidname",vidname)
	#<<<<<<< HEAD
        #print(index)
	#=======
        # print(index)
	#>>>>>>> ecd24e710dc9d4ce6715bb198bb5f0fba9a7f221
        random.shuffle(self.d[vidname])
        filenames = self.d[vidname][:1]
        self.id_num = self.vid2id[vidname]
        self.data_s = []
        self.ctx_frames_s = []
        self.main_fn_s = []
        for i in filenames:
            print("filename",i)
            self.load_data(i)
        # status_lst = Parallel(n_jobs=32,backend="threading")(delayed(self.load_data)(i) for i in filenames)            
        self.data_s, self.ctx_frames_s = (torch.stack(self.data_s).transpose(0,1),torch.stack(self.ctx_frames_s))
        return self.data_s, self.ctx_frames_s, self.main_fn_s,self.id_num

    def __len__(self):
        return len(self.id_names)

if __name__ == "__main__":
    train="data/train"
    train_mv="data/train_mv"
    from tempArg import TempArgument
    args = TempArgument(distance1=1,distance2=2,warp=True,v_compress=True,patch=64,num_crops=2,batch_size=1)
    print(args)
    # dset = ImageFolder(
    #     is_train=True,
    #     root=train,
    #     mv_dir=train_mv,
    #     args=args,
    # )
    train_loader = get_loader(is_train=True,root=args.train, mv_dir=args.train_mv,n_work=2,args=arg)

    # train_loader = get_loader(is_train=True,root=train, mv_dir=train_mv,n_work=0,args=args)

    for m,j,k,d in train_loader: 
        print("k")
    # from IPython import embed 
    # embed()



