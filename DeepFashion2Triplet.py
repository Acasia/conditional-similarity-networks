import os

num_attributeType = 5

attribute_name = list() # ["a-line", "abstract", ...]
attribute_type = list() # [0, 1, 0, 3, , ..] 총 attribute_name과 한쌍으로, attribute_name만큼 존재함

txtFile_names = ['filenames_train.txt', 'filenames_val.txt', 'filenames_test.txt']


for fileNames in txtFile_names:
    attributes = dict()  # attributes = 1 : [img1, img2, img3], 2 : [img1, img2, img3], 3 : [img1, img2, img3],..

    for num in range(1, num_attributeType + 1):
        attributes[num] = []

    print(attributes)
    '''
    train / val/ test file.txt를 순서대로 읽어옴
    '''
    with open(os.path.join('./../ASEN/data', fileNames)) as f:
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

    for i in range(1, 6): #attribute write file
        with open(os.path.join('./data/DeepFashion', fileNames.split('.')[0].split('_')[1] + 'att%s'%str(i) + '.txt'), 'w') as f:
            for item in attributes[i]:
                f.write("%s\n" % item)


# with open('./data/DeepFashion', 'r')