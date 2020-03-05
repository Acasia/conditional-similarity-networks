from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np


labelfiles = {
    'train': [
        '1_train.txt', #texture_related
        '2_train.txt', #fabric_related
        '3_train.txt', #shape_related
        '4_train.txt', #part_related
        '5_train.txt' #style_related
    ],
    'val': [
        '1_val.txt',
        '2_val.txt',
        '3_val.txt',
        '4_val.txt',
        '5_val.txt'
    ],
    'test': [
        '1_test.txt',
        '2_test.txt',
        '3_test.txt',
        '4_test.txt',
        '5_test.txt'
    ]
}

filenames = {
    'train': [
            'train_att1.txt',
            'train_att2.txt',
            'train_att3.txt',
            'train_att4.txt',
            'train_att5.txt'
        ],
    'val': [
            'val_att1.txt',
            'val_att2.txt',
            'val_att3.txt',
            'val_att4.txt',
            'val_att5.txt'
        ],
    'test': [
            'test_att1.txt',
            'test_att2.txt',
            'test_att3.txt',
            'test_att4.txt',
            'test_att5.txt'
        ]
}


def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, conditions, split, n_triplets, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers,
                where integer i refers to the i-th image in the filenames file.
                For a line of intergers 'a b c', a triplet is defined such that image a is more
                similar to image c than it is to image b, e.g.,
                0 2017 42 """
        self.root = root

        #모든 파일 경로가 적혀있는 txt. index로 접근가능함


        #fnames엔 split별 .txt파일 이름 5개가 들어감.
        triplets = []
        if split == 'train':
            fnames = filenames['train'] #20만개
            lnames = labelfiles['train']
        elif split == 'val':
            fnames = filenames['val'] #2만개
            lnames = labelfiles['val']
        else:
            fnames = filenames['test'] #4만개
            lnames = labelfiles['test']

        '''      
        train / val/ test별로 저 해당 파일의 index로 접근해서 가져오는 역할을 함
        '''
        attribute_dict = dict()  # conditions만큼 생성해서, 각 condition의 attribute별로 list로 파일 path를 저장했음

        for condition in conditions:  # a dictionary 초기화
            attribute_dict[condition] = []

        for condition in conditions:  # condition마다, 파일을 읽어서 attribute별로 list를 만들어서, a condition 위치에 추가함
            with open(os.path.join(self.root, 'DeepFashion', lnames[condition]), 'r') as f:  # 파일의 라인별로 읽어옴
                datas = f.readlines()
                for data in datas:
                    sub = []  # 임시 저장소
                    filepaths = data.split()[1:]  # condition의 한 attribute의 file path들
                    for filepath in filepaths:
                        sub.append(filepath.replace("'", "").replace("\n", "").replace("[", "").replace("]", "").replace(",", ""))
                    attribute_dict[condition].append(sub)

        for condition in conditions: #0, 1, 2, 3, 4
            '''
            해당 train/val/test별로 적혀있는 모든 txt를 돌면서 0, 1, 2번째, condition을 triplets에 append함.
            '''
            with open(os.path.join(self.root, 'DeepFashion', fnames[condition]), 'r') as f:
                datas = f.readlines()
                for anchor in datas:
                    anchor = anchor.replace("\n", "")

                    while True:
                        random_attribute_far = np.random.randint(len(attribute_dict[condition]))
                        random_imgIdx_far = np.random.randint(len(attribute_dict[condition][random_attribute_far]))
                        anchor_far = attribute_dict[condition][random_attribute_far][random_imgIdx_far]
                        # 다른 attribute img

                        if anchor != anchor_far and anchor not in attribute_dict[condition][random_attribute_far]:
                            # anchor_far와 같지 않고, attribute도 달라야 break
                            break

                    while True:
                        random_attribute_close = np.random.randint(len(attribute_dict[condition]))
                        random_imgIdx_close = np.random.randint(len(attribute_dict[condition][random_attribute_close]))
                        anchor_close = attribute_dict[condition][random_attribute_close][random_imgIdx_close]
                        # 같은 attribute img

                        if anchor != anchor_close and anchor in attribute_dict[condition][random_attribute_close]:
                            # anchor_close와 같지 않고, attribute도 같아야 break
                            break

                    triplets.append((anchor, anchor_far, anchor_close, condition)) # anchor, far, close, .txts


        np.random.shuffle(triplets)

        self.triplets = triplets[:int(n_triplets * 1.0 * len(conditions) / 4)]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3, c = self.triplets[index]
        '''
        *할일*
        index로 정의된 path1, 2, 3를 나는 이미 path로 정해두었기 때문에, 바로 self.loader로 열면 될 것 같다.
        '''
        if os.path.exists(os.path.join(self.root, path1)) and os.path.exists(os.path.join(self.root, path1)) and os.path.exists(os.path.join(self.root, path1)):
            img1 = self.loader(os.path.join(self.root, path1))
            img2 = self.loader(os.path.join(self.root, path2))
            img3 = self.loader(os.path.join(self.root, path3))
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return img1, img2, img3, c
        else:
            return None

    def __len__(self):
        return len(self.triplets)
