import torch
from torch.utils.data import Dataset
import random

from PIL import Image
import sys
import os
import numpy as np

from .utils import ImageUtilities as IU


class SegDataset(Dataset):
    """Dataset Reader"""
    # DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
    #                                         os.path.pardir, os.path.pardir))
    # SEMANTIC_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP',
    #                                 'semantic-annotations')
    # INSTANCE_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP',
    #                                 'instance-annotations')
    # IMG_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP', 'CVPPP2017_LSC_training',
    #                        'training', 'A1')
    # OUT_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP', 'lmdb')

    def __init__(self):

        self.semantic_ann_base_path = '/home/lab/Documents/식물/Code/instance-segmentation-pytorch/data/processed/CVPPP/semantic-annotations'
        self.instance_ann_base_path = '/home/lab/Documents/식물/Code/instance-segmentation-pytorch/data/processed/CVPPP/instance-annotations'
        self.img_base_path = '/home/lab/Documents/식물/Code/instance-segmentation-pytorch/data/raw/CVPPP/CVPPP2017_LSC_training/training/A1'
        self.n_samples = len(os.listdir(self.semantic_ann_base_path))

    def __load_data(self, index):
        if (index+1) < 10:
            name = 'plant00'
        elif (index +1) < 100:
            name = 'plant0'
        else:
            name = 'plant'

        name += str(index+1)

        img_path = os.path.join(self.img_base_path, (name + '_rgb.png'))
        img = Image.open(img_path)


        semantic_annotation = np.load(os.path.join(self.semantic_ann_base_path, name + '.npy'))
        instance_annotation = np.load(os.path.join(self.instance_ann_base_path, name + '.npy'))
        n_objects = instance_annotation.shape[2]

        return img, semantic_annotation, instance_annotation, n_objects

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        image, semantic_annotation, instance_annotation, n_objects = self.__load_data(index)

        return image, semantic_annotation, instance_annotation, n_objects

    def __len__(self):
        return self.n_samples


class AlignCollate(object):

    def __init__(self, mode, n_classes, max_n_objects, mean, std, image_height,image_width):

        self._mode = mode
        self.n_classes = n_classes
        self.max_n_objects = max_n_objects

        assert self._mode in ['training', 'test']

        self.mean = mean
        self.std = std
        self.image_height = image_height
        self.image_width = image_width

        self.random_horizontal_flipping = True
        self.random_vertical_flipping = True
        self.random_transposing =  True
        self.random_90x_rotation = True
        self.random_rotation = True


        if self._mode == 'training':
            if self.random_horizontal_flipping:
                self.horizontal_flipper = IU.image_random_horizontal_flipper()
            if self.random_vertical_flipping:
                self.vertical_flipper = IU.image_random_vertical_flipper()
            if self.random_transposing:
                self.transposer = IU.image_random_transposer()
            if self.random_rotation:
                self.image_rotator = IU.image_random_rotator(random_bg=True)
                self.annotation_rotator = IU.image_random_rotator(Image.NEAREST,
                                                                  random_bg=False)
            if self.random_90x_rotation:
                self.image_rotator_90x = IU.image_random_90x_rotator()
                self.annotation_rotator_90x = IU.image_random_90x_rotator(Image.NEAREST)


            self.img_resizer = IU.image_resizer(self.image_height,
                                                self.image_width)
            self.ann_resizer = IU.image_resizer(self.image_height,
                                                self.image_width,
                                                interpolation=Image.NEAREST)
        else:
            self.img_resizer = IU.image_resizer(self.image_height,
                                                self.image_width)
            self.ann_resizer = IU.image_resizer(self.image_height,
                                                self.image_width,
                                                interpolation=Image.NEAREST)

        self.image_normalizer = IU.image_normalizer(self.mean, self.std)

    def __preprocess(self, image, semantic_annotation, instance_annotation):

        # Augmentation
        if self._mode == 'training':
            instance_annotation = list(instance_annotation.transpose(2, 0, 1))
            n_objects = len(instance_annotation)



            if self.random_horizontal_flipping:
                is_flip = random.random() < 0.5
                image = self.horizontal_flipper(image, is_flip)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.horizontal_flipper(_ann, is_flip)
                    instance_annotation[i] = _ann

                semantic_annotation = self.horizontal_flipper(semantic_annotation, is_flip)

            if self.random_vertical_flipping:
                is_flip = random.random() < 0.5
                image = self.vertical_flipper(image, is_flip)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.vertical_flipper(_ann, is_flip)
                    instance_annotation[i] = _ann

                semantic_annotation = self.vertical_flipper(semantic_annotation, is_flip)

            if self.random_transposing:
                is_trans = random.random() < 0.5
                image = self.transposer(image, is_trans)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.transposer(_ann, is_trans)
                    instance_annotation[i] = _ann

                semantic_annotation = self.transposer(semantic_annotation, is_trans)

            if self.random_90x_rotation:
                rot_angle = np.random.choice([0, 90, 180, 270])
                rot_expand = True
                image = self.image_rotator_90x(image, rot_angle, rot_expand)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.annotation_rotator_90x(_ann, rot_angle, rot_expand)
                    instance_annotation[i] = _ann

                semantic_annotation = self.annotation_rotator_90x(semantic_annotation,rot_angle, rot_expand)

            if self.random_rotation:
                rot_angle = int(np.random.rand() * 10)
                if np.random.rand() >= 0.5:
                    rot_angle = -1 * rot_angle
                # rot_expand = np.random.rand() < 0.5
                rot_expand = True
                image = self.image_rotator(image, rot_angle, rot_expand)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.annotation_rotator(_ann, rot_angle, rot_expand)
                    instance_annotation[i] = _ann

                semantic_annotation = self.annotation_rotator(semantic_annotation,rot_angle, rot_expand)


            instance_annotation = np.array(instance_annotation).transpose(1, 2, 0)

        # Resize Images
        image = self.img_resizer(image)

        # Resize Instance Annotations
        ann_height, ann_width, n_objects = instance_annotation.shape
        instance_annotation_resized = []

        height_ratio = 1.0 * self.image_height / ann_height
        width_ratio = 1.0 * self.image_width / ann_width

        for i in range(n_objects):
            instance_ann_img = Image.fromarray(instance_annotation[:, :, i])
            instance_ann_img = self.ann_resizer(instance_ann_img)
            instance_ann_img = np.array(instance_ann_img)

            instance_annotation_resized.append(instance_ann_img)

        # Fill Instance Annotations with zeros
        for i in range(self.max_n_objects - n_objects):
            zero = np.zeros((ann_height, ann_width),dtype=np.uint8)
            zero = Image.fromarray(zero)
            zero = self.ann_resizer(zero)
            zero = np.array(zero)
            instance_annotation_resized.append(zero.copy())

        instance_annotation_resized = np.stack(instance_annotation_resized, axis=0)
        instance_annotation_resized = instance_annotation_resized.transpose(1, 2, 0)

        # Resize Semantic Anntations
        semantic_annotation = self.ann_resizer(Image.fromarray(semantic_annotation))
        semantic_annotation = np.array(semantic_annotation)

        # Image Normalization
        image = self.image_normalizer(image)

        return (image, semantic_annotation, instance_annotation_resized)

    def __call__(self, batch):  #객체 호출시 실행
        # (image, s_a, i_a) 형태로 있는걸 image 끼리, s_a 끼리, i_n끼리 묶는
        images, semantic_annotations, instance_annotations, n_objects = zip(*batch)

        images = list(images)
        semantic_annotations = list(semantic_annotations)
        instance_annotations = list(instance_annotations)

        bs = len(images)
        for i in range(bs):

            #augmentation 어어
            self.random_horizontal_flipping = np.random.choice([True, False])
            self.random_horizontal_flipping = np.random.choice([True, False])
            self.random_vertical_flipping = np.random.choice([True, False])
            self.random_transposing = np.random.choice([True, False])
            self.random_90x_rotation = np.random.choice([True, False])
            self.random_rotation = np.random.choice([True, False])

            image, semantic_annotation, instance_annotation = \
                self.__preprocess(images[i],semantic_annotations[i],instance_annotations[i])

            images[i] = image
            semantic_annotations[i] = semantic_annotation
            instance_annotations[i] = instance_annotation

        images = torch.stack(images)

        instance_annotations = np.array(instance_annotations,dtype='int')  # bs, h, w, n_ins

        semantic_annotations = np.array(semantic_annotations, dtype='int')  # bs, h, w
        semantic_annotations_one_hot = np.eye(self.n_classes, dtype='int')
        semantic_annotations_one_hot = \
            semantic_annotations_one_hot[semantic_annotations.flatten()].reshape(
                semantic_annotations.shape[0], semantic_annotations.shape[1],
                semantic_annotations.shape[2], self.n_classes)

        instance_annotations = torch.LongTensor(instance_annotations)
        instance_annotations = instance_annotations.permute(0, 3, 1, 2)

        semantic_annotations_one_hot = torch.LongTensor(semantic_annotations_one_hot)
        semantic_annotations_one_hot = semantic_annotations_one_hot.permute(0, 3, 1, 2)

        n_objects = torch.IntTensor(n_objects)

        return (images, semantic_annotations_one_hot, instance_annotations,n_objects)

#직접 실행시켰을 때만 실행되기를 원하는 코
if __name__ == '__main__':
    ds = SegDataset()
    image, semantic_annotation, instance_annotation, n_objects = ds[5]

    print(image.size)
    print(semantic_annotation.shape)
    print(instance_annotation.shape)
    print(n_objects)
    print(np.unique(semantic_annotation))
    print(np.unique(instance_annotation))

    ac = AlignCollate('training', 9, 120, [0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0], 256, 512)

    loader = torch.utils.data.DataLoader(ds, batch_size=3,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=False,
                                         collate_fn=ac)
    loader = iter(loader)

    images, semantic_annotations, instance_annotations, n_objects = loader.next()

    print(images.size())
    print(semantic_annotations.size())
    print(instance_annotations.size())
    print(n_objects.size())
    print(n_objects)
