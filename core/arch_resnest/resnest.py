##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from .resnet import ResNet, Bottleneck

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://github-releases.githubusercontent.com/168799526/c5627180-ba4e-11ea-95f5-31e2a98de152?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210628T030939Z&X-Amz-Expires=300&X-Amz-Signature=d9f3082152727f717bb46a32e9663759ee3f6efbd52347f3b94c4d7f33bc9216&X-Amz-SignedHeaders=host&actor_id=42644052&key_id=0&repo_id=168799526&response-content-disposition=attachment%3B%20filename%3Dresnest50-528c19ca.pth&response-content-type=application%2Foctet-stream", progress=True, check_hash=True))

    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        # model.load_state_dict(torch.hub.load_state_dict_from_url(
        #     resnest_model_urls['resnest101'], progress=True, check_hash=True))
        model.load_state_dict(torch.load("/home/ders/.cache/torch/hub/checkpoints/resnest101-22405ba7.pth"))

    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://github-releases.githubusercontent.com/247528876/3b82cc00-b726-11eb-81ae-8935a038a777?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210628T061940Z&X-Amz-Expires=300&X-Amz-Signature=71a1ecfb86a342df4576c69137780d0cb865b42eda9c2f17e544f59c388964e9&X-Amz-SignedHeaders=host&actor_id=42644052&key_id=0&repo_id=247528876&response-content-disposition=attachment%3B%20filename%3Dresnest269-0cc87c48.pth&response-content-type=application%2Foctet-stream", progress=True, check_hash=True))
    return model
