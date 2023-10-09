from typing import List
import os
import datetime
import numpy as np
from PIL import Image
import kornia
import torch
from einops import rearrange

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
from video_diffusion.common.image_util import save_gif_mp4_folder_type
from video_diffusion.prompt_attention.attention_store import AttentionStore
import torchvision.transforms as transforms
from einops import rearrange
from kornia.morphology import dilation

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.dim() == 3:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
            elif item.dim() == 4:
                t, h, res_sq, token = item.shape
                if item.shape[2] == num_pixels:
                    # only take the 256 part
                    cross_maps = item.reshape(len(prompts), t, -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
                    
    out = torch.cat(out, dim=-4)
    out = out.sum(-4) / out.shape[-4]
    return out.cpu()


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0, save_path = None,editing_type = 'attribute',dilation_kernel = None):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
        select is for choosing prompt
    """
    if isinstance(prompts, str):
        prompts = [prompts,]
    tokens = tokenizer.encode(prompts[select]) 
    decoder = tokenizer.decode
    
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    os.makedirs('trash', exist_ok=True)
    attention_list = []
    if attention_maps.dim()==3: attention_maps=attention_maps[None, ...]
    if editing_type == 'attribute' or editing_type == 'background':
        is_dilation = True
        mask_threshold = 100
        kernel = torch.ones(3,3)
        if dilation_kernel is not None:
            kernel = torch.ones(dilation_kernel,dilation_kernel)
    # elif editing_type == 'shape':
    #     is_dilation = True
    #     mask_threshold = 100
    #     kernel = torch.ones(3,3)
    elif editing_type == 'style':
        mask_threshold = 0

    for j in range(attention_maps.shape[0]): 
        images = []
        for i in range(len(tokens)):
            # token i 77
            image = attention_maps[j, :, :, i]
            # print('image',image.shape)([16, 16])
            image = 255 * image / image.max()
            image = torch.where(image < mask_threshold,torch.tensor(0), torch.tensor(255))
            if is_dilation:
                image = dilation(image[None,None,],kernel)[0][0]
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            # print('image',image.shape)([16, 16, 3])
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((512,512)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path)
        atten_j = np.concatenate(images, axis=1)
        attention_list.append(atten_j)
    # if save_path is not None:
    #     now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    #     video_save_path = f'{save_path}/{now}.gif'
        # save_gif_mp4_folder_type(attention_list, video_save_path)
    return attention_list

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def get_attention_mask(tokenizer, prompts, edit_phrase, attention_store: AttentionStore, 
                         res: int, from_where: List[str], select: int = 0,editing_type = 'attribute',dilation_kernel = None):
    """
        attention_store (AttentionStore): 
            ["down", "mid", "up"] X ["self", "cross"]
            4,         1,    6
            head*res*text_token_len = 8*res*77
            res=1024 -> 64 -> 1024
        res (int): res
        from_where (List[str]): "up", "down'
        select is for choosing prompt
    """
    if isinstance(prompts, str):
        prompts = [prompts,]
    if isinstance(edit_phrase, str):
        edit_phrase = [edit_phrase,]
    tokens = tokenizer.encode(prompts[select]) 
    edit_phrase_tokens = tokenizer.encode(edit_phrase[select])[1:-1]
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    os.makedirs('trash', exist_ok=True)
    mask_list = []
    if editing_type == 'attribute' or editing_type == 'background':
        is_dilation = True
        mask_threshold = 100
        kernel = torch.ones(3,3)
        if dilation_kernel is not None:
            kernel = torch.ones(dilation_kernel,dilation_kernel)
    # elif editing_type == 'shape':
    #     is_dilation = True
    #     mask_threshold = 100
    #     kernel = torch.ones(3,3)
    elif editing_type == 'style':
        mask_threshold = 0


    if attention_maps.dim()==3: attention_maps=attention_maps[None, ...]
    for j in range(attention_maps.shape[0]):
        # frame j
        images = []
        for i in range(len(tokens)):
            # token i num of words
            if tokens[i] in edit_phrase_tokens:
                image = attention_maps[j, :, :, i]
                image = 255 * image / image.max()

                image = torch.where(image < mask_threshold,torch.tensor(0), torch.tensor(1))
                if is_dilation:
                    image = dilation(image[None,None,],kernel)[0][0]

                image = image.numpy().astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((64,64)))
                images.append(torch.from_numpy(image))

        # different frame mask stack
        frame_mask, _ = torch.max(torch.stack(images),dim = 0)
        mask_list.append(frame_mask)

    mask = torch.stack(mask_list,dim = 0)
    return mask[None,None,]

