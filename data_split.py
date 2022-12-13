import splitfolders
import os
import yaml
import shutil

def read_yaml(path):
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        return data

def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Error: Creating directory. ' +  path)

if __name__ == '__main__':
    data_path = '224x224'
    save_path = 'images'
    label_list = []
    num = 0

    dataset_info = read_yaml('bookcover_class.yaml')
    print("number of class: {0}".format(dataset_info['class']))
    f = open("label.txt", 'r')

    while True:
        line = f.readline()
        if not line: break
        image, cls = line.split(' ')
        cls = cls.strip()
        data = [image, int(cls)]
        label_list.append(data)

    f.close()
    for el in label_list:
        image_name = el[0]
        image_class_idx = el[1]
        image_class = dataset_info['class'][image_class_idx]
        image_path = os.path.join(data_path, image_name)
        class_path = os.path.join(save_path, image_class)
        make_dir(class_path)

        img_rename = str(num).zfill(5) + '.jpg'
        img_re_path = os.path.join(class_path, img_rename)
        shutil.copyfile(image_path, img_re_path)
        num += 1
        print('{0}/{1}'.format(num, len(label_list)))

    splitfolders.ratio(save_path, output='split_images', seed=77, ratio=(.8, 0.1, 0.1))










