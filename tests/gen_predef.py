import torch
import torch.nn as nn

from typing import List, Callable
from random import random
from hashlib import sha256
from os.path import join

from models import ResConvBlock

import inspect


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


def v001():
    name = globals()[inspect.getframeinfo(inspect.currentframe()).function].__name__
    id_code = sha256(name.encode('utf-8')).hexdigest()[:10]

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
    id_code = sha256(name.encode('utf-8')).hexdigest()[:10]

    model = ResConvBlock(6, 10, is_down=True, use_res=False)
    model.eval()
    saver(inputs=[torch.randn(4, 6, 256, 256)],
          model=model,
          in_map_fn=lambda m, x: m(x[0]),
          out_map_fn=lambda x: [x],
          id_code=id_code,
          )


def main():
    v001()
    v002()


if __name__ == '__main__':
    main()
