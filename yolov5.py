import argparse
import subprocess
# import os
import glob
# from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', help='for train command')
parser.add_argument('--val', action='store_true', help='for val command')
parser.add_argument('--test', action='store_true', help='for test command')

# train, val option
parser.add_argument('--batch_size', '--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--img', '--img-size', default=1200, help='inference size (pixels)')
parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
parser.add_argument('--device', default='', help='cuda device, i,e, 0 or 0,1,2,3 or cpu')
parser.add_argument('--weights', default=sorted(glob.glob('ckpt/*.pt'))[0], help='cuda device, i,e, 0 or 0,1,2,3 or cpu')

args = parser.parse_args()

data = 'cfg/nia50_data_yolov5l6.yaml'
model = 'cfg/nia50_model_yolov5l6.yaml'
# weights = 'ckpt/nia50_bestweights_yolov5l6.pt'
# weights = sorted(os.listdir('ckpt'))[0]
weights = args.weights
project = 'result/nia50'

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
    global project

    if args.train:
        img = int(args.img)
        batch_size = args.batch_size
        epochs = args.epochs
        if args.device: device = args.device
        else: device = 0
        optimizer = 'AdamW'
        # project = 'result/nia50'
        name = 'train'
        run_train = f'python3 model/train.py --img {img} --batch-size {batch_size} --epochs {epochs} --optimizer {optimizer} \
        --project {project} --name {name} --data {data} --cfg {model} --device {device}'
        subprocess.call(run_train, shell=True)
    
    if args.val:
        img = int(args.img)
        batch_size = args.batch_size
        if args.device: device = args.device
        else: device = 0
        # project = 'result/nia50'
        name = 'val'
        run_validate = f'python3 model/val.py --img {img} --batch-size {batch_size} --name {name} --project {project} --data {data} --weights {weights} \
        --verbose --save-txt --save-conf --save-json --exist-ok --device {device}'
        subprocess.call(run_validate, shell=True)

    if args.test:
        img = str(args.img)
        # project = 'result/nia50'
        name = 'test'
        # source = cwd + '/data/images'
        source = '../Data/50-2/images_2d_test'
        run_test = f'python3 model/detect.py --img {img} --conf 0.4 --project {project} --name {name} --source {source} --weights {weights} \
        --save-txt --save-conf --exist-ok'
        subprocess.call(run_test, shell=True)

if __name__ == '__main__':
    main()