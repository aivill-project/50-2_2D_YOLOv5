import argparse
import subprocess
import os
import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', help='for train command')
parser.add_argument('--val', action='store_true', help='for val command')
parser.add_argument('--test', action='store_true', help='for test command')

# train, val option
parser.add_argument('--batch_size', '--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--img', '--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
parser.add_argument('--device', default='', help='cuda device, i,e, 0 or 0,1,2,3 or cpu')

args = parser.parse_args()

data = 'cfg/nia50_data_yolov5l6.yaml'
model = 'cfg/nia50_model_yolov5l6.yaml'
weights = 'ckpt/nia50_bestweights_yolov5l6.pt'

def main():
    # cwd = os.getcwd()
    # images = glob.glob(cwd+'/data/images/*.jpg')
    # train, val = train_test_split(images, test_size=0.2, random_state=0)

    # with open(cwd+'/data/ImageSets/train.txt', 'w') as f:
    #     f.write('\n'.join(train))
    
    # with open(cwd+'/data/ImageSets/val.txt', 'w') as f:
    #     f.write('\n'.join(val))
    
    global data
    global model
    global weights

    if args.train:
        img = args.img
        batch_size = args.batch_size
        epochs = args.epochs
        device = args.device
        optimizer = 'AdamW'
        project = 'result/train'
        name = 'nia50'
        run_train = f'python3 model/train.py --img {img} --batch-size {batch_size} --epochs {epochs} --optimizer {optimizer} \
        --project {project} --name {name} --data {data} --cfg {model} --weights {weights} --device {device}'
        subprocess.call(run_train, shell=True)
    
    if args.val:
        img = args.img
        batch_size = args.batch_size
        device = args.device
        project = 'result/val'
        name = 'nia50'
        run_validate = f'python3 model/val.py --img {img} --batch-size {batch_size} --name {name} --project {project} --data {data} --weights {weights} \
        --verbose --save-txt --save-conf --save-json --exist-ok --device {device}'
        subprocess.call(run_validate, shell=True)

    if args.test:
        img = '1200 1920'
        project = 'result/test'
        name = 'nia50'
        # source = cwd + '/data/images'
        source = '../Data/50-2/images_2d'
        run_test = f'python3 model/detect.py --img {img} --conf 0.4 --project {project} --name {name} --source {source} --weights {weights} \
        --save-txt --save-conf --exist-ok'
        subprocess.call(run_test, shell=True)

if __name__ == '__main__':
    main()