import os

def deepfashion2Attribute():
    num_attributeType = 5

    attribute_name = list() # ["a-line", "abstract", ...]
    attribute_type = list() # [0, 1, 0, 3, , ..] 총 attribute_name과 한쌍으로, attribute_name만큼 존재함

    txtFile_names = ['filenames_train.txt', 'filenames_val.txt', 'filenames_test.txt']

    for fileNames in txtFile_names:
        attributes = dict()  # attributes = 1 : [img1, img2, img3], 2 : [img1, img2, img3], 3 : [img1, img2, img3],..
        sub_attributes = dict()

        '''
        sub_attributes = 
        1 : {'aline' : [img1, img2, img3], 'hline' :[img1, img2, img3], ..}
        2 : {'가죽'  : [img1, img2, img3], '면'    :[img1, img2, img3], ..}, ..
        '''
        for num in range(1, num_attributeType + 1):
            attributes[num] = []
            sub_attributes[num] = dict()

        '''
        train / val/ test file.txt를 순서대로 읽어옴
        '''
        with open(os.path.join('./../ASEN/data', fileNames), 'r') as f:
            mode_img_names = f.readlines() #filepath 들이 적혀있음.

        for i in range(0, len(mode_img_names)-1):
            mode_img_names[i] = mode_img_names[i].replace("\n" ,"")

        with open('./../ASEN/data/list_attr_cloth.txt', 'r') as f:
            attribute_name_lines = f.readlines() #attribute_name, attribute_type들이 적혀있음

        num_attrs = int(attribute_name_lines.pop(0).strip())  # attribute 개수(1000)
        header_attr = attribute_name_lines.pop(0).split()  # 'attribute_name', 'attribute_type'

        for element in attribute_name_lines:
            columns = element.split()  # attribute_name, attribute_type

            attribute_name.append('-'.join(columns[:-1])) # attribute name
            attribute_type.append(columns[-1])  # attribute type

            #if 'key1' in dict.keys():
            for num in range(1, num_attributeType + 1):
                if '-'.join(columns[:-1]) not in sub_attributes[int(columns[-1])].keys():
                    sub_attributes[int(columns[-1])]['-'.join(columns[:-1])] = []

        with open('./../ASEN/data/list_attr_img.txt', 'r') as f:
            img_labels = f.readlines()

        num_images = int(img_labels.pop(0).strip())  # image 개수(289222)
        header_label = img_labels.pop(0).split()  # 'image_name', 'attribute_labels'

        for element in img_labels:
            columns = element.split()  # filename, label(-1 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1)
            filename = columns[0]  # filename

            if filename in mode_img_names:  # mode에 맞는 label만 선택하여 추가
                for x, v in enumerate(columns[1:]):
                    if int(v) == 1:
                        if  filename not in attributes[int(attribute_type[int(x)])]:
                            attributes[int(attribute_type[int(x)])].append(filename)

                        '''
                        0번째 attribute type을 key로 접근해서 내부 dictionary에 접근함.
                        해당 attribute name을 key로 설정해서 내부 dictionary value에 filename을 추가하는 코드
                        '''
                        if filename not in sub_attributes[int(attribute_type[int(x)])][attribute_name[int(x)]]:
                            sub_attributes[int(attribute_type[int(x)])][attribute_name[int(x)]].append(filename)

        for i in range(1, 6): #attribute write file
            with open(os.path.join('./data/DeepFashion', fileNames.split('.')[0].split('_')[1] + '_att%s'%str(i) + '.txt'), 'w') as f:
                for item in attributes[i]:
                    f.write("%s\n" % item)

        '''
        0_train_.txt - 
            aline filename filename filename
            bline filename filename filename
        0_val_.txt
            aline filename filename filename
            bline filename filename filename
        1_train_.txt -
            가죽 filename filename filename
            면 filename filename filename
        '''
        for i in range(1, 6): #attribute write file
            with open(os.path.join('./data/DeepFashion', '%s_'%str(i) + fileNames.split('.')[0].split('_')[1]  + '.txt'), 'w') as f:
                for items in sub_attributes[i]:
                    f.write("%s " % items)
                    for item in sub_attributes[i][items]:
                        f.write('%s ' % item)
                    f.write("\n")

deepfashion2Attribute()