import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')

args = parser.parse_args()

def test():
    print('no')

    if args.test:
        print('yes')

if __name__ == '__main__':
    if args.test:
        print('test ok')
# print("test")