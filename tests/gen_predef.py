import torch
import torch.nn as nn

from typing import List, Callable
from hashlib import sha256
from os.path import join

from models import (ResConvBlock, EncoderF_Res)
from global_config import LEN_HASH

import inspect
import argparse

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('targets', type=str, nargs='+')

    return p.parse_args()



def v001():
    name = globals()[inspect.getframeinfo(inspect.currentframe()).function].__name__
    print(name, 'started')
    id_code = sha256(name.encode('utf-8')).hexdigest()[:LEN_HASH]

    model = ResConvBlock(6, 10, is_down=True)
    model.eval()
    saver(inputs=[torch.randn(4, 6, 256, 256)],
          model=model,
          in_map_fn=lambda m, x: m(x[0]),
          out_map_fn=lambda x: [x],
          id_code=id_code,
          )
    

def v002():
    name = globals()[inspect.getframeinfo(inspect.currentframe()).function].__name__
    print(name, 'started')
    id_code = sha256(name.encode('utf-8')).hexdigest()[:LEN_HASH]

    model = ResConvBlock(6, 10, is_down=True, use_res=False)
    model.eval()
    saver(inputs=[torch.randn(4, 6, 256, 256)],
          model=model,
          in_map_fn=lambda m, x: m(x[0]),
          out_map_fn=lambda x: [x],
          id_code=id_code,
          )


def v003():
    name = globals()[inspect.getframeinfo(inspect.currentframe()).function].__name__
    print(name, 'started')
    id_code = sha256(name.encode('utf-8')).hexdigest()[:LEN_HASH]

    model = EncoderF_Res()
    model.eval()

    saver(inputs=[torch.randn(4, 1, 256, 256)],
          model=model,
          in_map_fn=lambda m, x: m(x[0]),
          out_map_fn=lambda x: [x],
          id_code=id_code,
          )


def v004():
    name = globals()[inspect.getframeinfo(inspect.currentframe()).function].__name__
    print(name, 'started')
    id_code = sha256(name.encode('utf-8')).hexdigest()[:LEN_HASH]

    model = EncoderF_Res(norm='adabatch')
    model.eval()

    saver(inputs=[torch.randn(4, 1, 256, 256), torch.randn(4, 128)],
          model=model,
          in_map_fn=lambda m, x: m(x[0],x[1]),
          out_map_fn=lambda x: [x],
          id_code=id_code,
          )


def saver(inputs: List[torch.Tensor],
          model: nn.Module,
          in_map_fn: Callable,
          out_map_fn: Callable,
          id_code: str,
          path_dir='tests/storage'
          ):
    name = type(model).__name__

    path = '%s_m_%s' % (id_code, name)
    path = join(path_dir, path)
    torch.save(model.state_dict(), path)

    outputs = in_map_fn(model, inputs)
    outputs: List[torch.Tensor] = out_map_fn(outputs)

    for i, input in enumerate(inputs):
        path = '%s_input_%01d' % (id_code, i)
        path = join(path_dir, path)
        torch.save(input, path)

    for i, output in enumerate(outputs):
        path = '%s_output_%01d' % (id_code, i)
        path = join(path_dir, path)
        torch.save(output, path)


def main():
    args = parse()

    for target in args.targets:
        globals()[target]()


if __name__ == '__main__':
    main()
