import sys
import struct
import numpy as np

def load_cprd(path):
    with open(path, 'rb') as f:
        ba = bytearray(f.read())
    assert ba[0:4] == b'CPRD', 'not a CPRD file'
    ba = ba[4:]
    b   = struct.unpack('I', ba[0:4])[0]      # batch size (number of images)
    cls = struct.unpack('I', ba[4:8])[0]      # number of classes
    ba = ba[8:]
    classes = list(struct.unpack(f'{b}I', ba[:4*b]));           ba = ba[4*b:]
    prob    = list(struct.unpack(f'{b}f', ba[:4*b]));           ba = ba[4*b:]
    prob_matrix = np.array(struct.unpack(f'{b*cls}f', ba[:4*b*cls])).reshape(b, cls)
    return b, cls, classes, prob, prob_matrix

if __name__ == '__main__':
    path = sys.argv[1]
    b, cls, classes, prob, pm = load_cprd(path)
    print(f'{path}: batch={b} images, {cls} classes')
    print(f'softmax rows sum to ~1: {pm.sum(axis=1)[:min(b,5)]}')
    for i in range(b):
        top5 = np.argsort(pm[i])[::-1][:5]
        tops = ', '.join(f'#{c}={pm[i,c]:.3f}' for c in top5)
        print(f'  img[{i}]: predicted class {classes[i]:3d} (p={prob[i]:.4f})  | top5: {tops}')
