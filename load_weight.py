import torch
import re
def load_lama_to_controlnet(model, lama_ckpt_path, verbose=True):
    ckpt = torch.load(lama_ckpt_path, map_location='cpu', weights_only=False)

    # 兼容不同存储格式
    if 'state_dict' in ckpt:
        pretrained_dict = ckpt['state_dict']
    else:
        pretrained_dict = ckpt

    model_dict = model.state_dict()

    for k, v in pretrained_dict.items():
        if k == 'generator.model.1.ffc.convl2l.weight':
            print(1)
            model_dict['model.0.1.ffc.convl2l.weight'] = v
        if k == 'generator.model.2.ffc.convl2l.weight':
            model_dict['model.1.ffc.convl2l.weight'] = v
        if k == 'generator.model.3.ffc.convl2l.weight':
            model_dict['model.2.ffc.convl2l.weight'] = v
        if k == 'generator.model.4.ffc.convl2l.weight':
            model_dict['model.3.ffc.convl2l.weight'] = v
        if k == 'generator.model.4.ffc.convl2g.weight':
            model_dict['model.3.ffc.convl2g.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convl2l.weight':
            model_dict['model.4.conv1.ffc.convl2l.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convl2g.weight':
            model_dict['model.4.conv1.ffc.convl2g.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convg2l.weight':
            model_dict['model.4.conv1.ffc.convg2l.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convg2g.conv1.0.weight':
            model_dict['model.4.conv1.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convg2g.conv1.1.weight':
            model_dict['model.4.conv1.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convg2g.conv1.1.bias':
            model_dict['model.4.conv1.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.5.conv1.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.4.conv1.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.5.conv1.ffc.convg2g.conv2.weight':
            model_dict['model.4.conv1.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convl2l.weight':
            model_dict['model.4.conv2.ffc.convl2l.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convl2g.weight':
            model_dict['model.4.conv2.ffc.convl2g.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convg2l.weight':
            model_dict['model.4.conv2.ffc.convg2l.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convg2g.conv1.0.weight':
            model_dict['model.4.conv2.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convg2g.conv1.1.weight':
            model_dict['model.4.conv2.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convg2g.conv1.1.bias':
            model_dict['model.4.conv2.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.5.conv2.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.4.conv2.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.5.conv2.ffc.convg2g.conv2.weight':
            model_dict['model.4.conv2.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.6.conv1.ffc.convl2l.weight':
            model_dict['model.5.0.conv1.ffc.convl2l.weight'] = v
        if k == 'enerator.model.6.conv1.ffc.convl2g.weight':
            model_dict['model.5.0.conv1.ffc.convl2g.weight'] = v
        if k == 'generator.model.6.conv1.ffc.convg2l.weight':
            model_dict['model.5.0.conv1.ffc.convg2l.weight'] = v
        if k == 'generator.model.6.conv1.ffc.convg2g.conv1.0.weight':
            model_dict['model.5.0.conv1.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.6.conv1.ffc.convg2g.conv1.1.weight':
            model_dict['model.5.0.conv1.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.6.conv1.ffc.convg2g.conv1.1.bias':
            model_dict['model.5.0.conv1.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.6.conv1.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.5.0.conv1.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.6.conv1.ffc.convg2g.conv2.weight':
            model_dict['model.5.0.conv1.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convl2l.weight':
            model_dict['model.5.0.conv2.ffc.convl2l.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convl2g.weight':
            model_dict['model.5.0.conv2.ffc.convl2g.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convg2l.weight':
            model_dict['model.5.0.conv2.ffc.convg2l.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convg2g.conv1.0.weight':
            model_dict['model.5.0.conv2.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convg2g.conv1.1.weight':
            model_dict['model.5.0.conv2.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convg2g.conv1.1.bias':
            model_dict['model.5.0.conv2.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.6.conv2.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.5.0.conv2.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.6.conv2.ffc.convg2g.conv2.weight':
            model_dict['model.5.0.conv2.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convl2l.weight':
            model_dict['model.5.1.conv1.ffc.convl2l.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convl2g.weight':
            model_dict['model.5.1.conv1.ffc.convl2g.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convg2l.weight':
            model_dict['model.5.1.conv1.ffc.convg2l.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convg2g.conv1.0.weight':
            model_dict['model.5.1.conv1.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convg2g.conv1.1.weight':
            model_dict['model.5.1.conv1.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convg2g.conv1.1.bias':
            model_dict['model.5.1.conv1.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.7.conv1.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.5.1.conv1.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.7.conv1.ffc.convg2g.conv2.weight':
            model_dict['model.5.1.conv1.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convl2l.weight':
            model_dict['model.5.1.conv2.ffc.convl2l.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convl2g.weight':
            model_dict['model.5.1.conv2.ffc.convl2g.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convg2l.weight':
            model_dict['model.5.1.conv2.ffc.convg2l.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convg2g.conv1.0.weight':
            model_dict['model.5.1.conv2.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convg2g.conv1.1.weight':
            model_dict['model.5.1.conv2.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convg2g.conv1.1.bias':
            model_dict['model.5.1.conv2.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.7.conv2.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.5.1.conv2.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.7.conv2.ffc.convg2g.conv2.weight':
            model_dict['model.5.1.conv2.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convl2l.weight':
            model_dict['model.6.2.conv1.ffc.convl2l.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convl2g.weight':
            model_dict['model.6.2.conv1.ffc.convl2g.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convg2l.weight':
            model_dict['model.6.2.conv1.ffc.convg2l.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convg2g.conv1.0.weight':
            model_dict['model.6.2.conv1.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convg2g.conv1.1.weight':
            model_dict['model.6.2.conv1.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convg2g.conv1.1.bias':
            model_dict['model.6.2.conv1.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.8.conv1.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.6.2.conv1.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.8.conv1.ffc.convg2g.conv2.weight':
            model_dict['model.6.2.conv1.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convl2l.weight':
            model_dict['model.6.2.conv2.ffc.convl2l.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convl2g.weight':
            model_dict['model.6.2.conv2.ffc.convl2g.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convg2l.weight':
            model_dict['model.6.2.conv2.ffc.convg2l.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convg2g.conv1.0.weight':
            model_dict['model.6.2.conv2.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convg2g.conv1.1.weight':
            model_dict['model.6.2.conv2.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convg2g.conv1.1.bias':
            model_dict['model.6.2.conv2.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.8.conv2.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.6.2.conv2.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.8.conv2.ffc.convg2g.conv2.weight':
            model_dict['model.6.2.conv2.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convl2l.weight':
            model_dict['model.6.3.conv1.ffc.convl2l.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convl2g.weight':
            model_dict['model.6.3.conv1.ffc.convl2g.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convg2l.weight':
            model_dict['model.6.3.conv1.ffc.convg2l.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convg2g.conv1.0.weight':
            model_dict['model.6.3.conv1.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convg2g.conv1.1.weight':
            model_dict['model.6.3.conv1.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convg2g.conv1.1.bias':
            model_dict['model.6.3.conv1.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.9.conv1.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.6.3.conv1.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.9.conv1.ffc.convg2g.conv2.weight':
            model_dict['model.6.3.conv1.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convl2l.weight':
            model_dict['model.6.3.conv2.ffc.convl2l.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convl2g.weight':
            model_dict['model.6.3.conv2.ffc.convl2g.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convg2l.weight':
            model_dict['model.6.3.conv2.ffc.convg2l.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convg2g.conv1.0.weight':
            model_dict['model.6.3.conv2.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convg2g.conv1.1.weight':
            model_dict['model.6.3.conv2.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convg2g.conv1.1.bias':
            model_dict['model.6.3.conv2.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.9.conv2.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.6.3.conv2.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.9.conv2.ffc.convg2g.conv2.weight':
            model_dict['model.6.3.conv2.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convl2l.weight':
            model_dict['model.7.4.conv1.ffc.convl2l.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convl2g.weight':
            model_dict['model.7.4.conv1.ffc.convl2g.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convg2l.weight':
            model_dict['model.7.4.conv1.ffc.convg2l.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convg2g.conv1.0.weight':
            print(1)
            model_dict['model.7.4.conv1.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convg2g.conv1.1.weight':
            model_dict['model.7.4.conv1.ffc.convg2g.conv1.1.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convg2g.conv1.1.bias':
            model_dict['model.7.4.conv1.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.10.conv1.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.7.4.conv1.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.10.conv1.ffc.convg2g.conv2.weight':
            model_dict['model.7.4.conv1.ffc.convg2g.conv2.weight'] = v
        if k == 'generator.model.10.conv2.ffc.convl2l.weight':
            model_dict['model.7.4.conv2.ffc.convl2l.weight'] = v
        if k == 'generator.model.10.conv2.ffc.convl2g.weight':
            model_dict['model.7.4.conv2.ffc.convl2g.weight'] = v
        if k == 'generator.model.10.conv2.ffc.convg2l.weight':
            model_dict['model.7.4.conv2.ffc.convg2l.weight'] = v
        if k == 'generator.model.10.conv2.ffc.convg2g.conv1.0.weight':
            model_dict['model.7.4.conv2.ffc.convg2g.conv1.0.weight'] = v
        if k == 'generator.model.10.conv2.ffc.convg2g.conv1.1.weight':
            model_dict['model.7.4.conv2.ffc.convg2g.conv1.1.weight'] = v
            print(1)
        if k == 'generator.model.10.conv2.ffc.convg2g.conv1.1.bias':
            model_dict['model.7.4.conv2.ffc.convg2g.conv1.1.bias'] = v
        if k == 'generator.model.10.conv2.ffc.convg2g.fu.conv_layer.weight':
            model_dict['model.7.4.conv2.ffc.convg2g.fu.conv_layer.weight'] = v
        if k == 'generator.model.10.conv2.ffc.convg2g.conv2.weight':
            model_dict['model.7.4.conv2.ffc.convg2g.conv2.weight'] = v

    model.load_state_dict(model_dict, strict=False)

    return model

def load_controlnet(model, lama_ckpt_path, verbose=True):
    ckpt = torch.load(lama_ckpt_path, map_location='cpu', weights_only=False)

    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    model_dict = model.state_dict()

    loaded_keys = []
    skipped_keys = []

    for k, v in pretrained_dict.items():
        if k not in model_dict:
            skipped_keys.append((k, 'not in model'))
            continue

        if re.search(r"(bn|batchnorm|running_mean|running_var)", k, re.I):
            skipped_keys.append((k, 'bn / running buffer'))
            continue

        if model_dict[k].shape == v.shape:
            model_dict[k] = v
            loaded_keys.append(k)
            continue

        skipped_keys.append((k, f"shape mismatch {v.shape} vs {model_dict[k].shape}"))

    model.load_state_dict(model_dict, strict=False)

    if verbose:
        print("=" * 50)
        print(f"Loaded layers: {len(loaded_keys)}")
        print(f"Skipped layers: {len(skipped_keys)}")

        print("\n--- Loaded sample ---")
        print(loaded_keys[:10])

        print("\n--- Skipped sample ---")
        for s in skipped_keys[:10]:
            print(s)

    return model

def freeze_lama_part(model):
    for name, param in model.named_parameters():
        if any(x in name for x in [
            'time_embed',
            'emb_layers',
            'input_hint_block',
            'zero_convs'
        ]):
            param.requires_grad = True
        else:
            param.requires_grad = False