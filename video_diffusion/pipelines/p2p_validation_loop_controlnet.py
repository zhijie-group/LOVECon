import os
import numpy as  np
from typing import List, Union
import PIL
import copy
from einops import rearrange

import torch
import torch.utils.data
import torch.utils.checkpoint

from diffusers.pipeline_utils import DiffusionPipeline
from tqdm.auto import tqdm
from video_diffusion.common.image_util import make_grid, annotate_image
from video_diffusion.common.image_util import save_gif_mp4_folder_type
from PIL import Image
from torch.nn import functional as F
from RIFEModel.RIFE_HDv3 import Model

class P2pSampleLogger:
    def __init__(
        self,
        editing_prompts: List[str],
        clip_length: int,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        sample_seeds: List[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7,
        strength: float = None,
        annotate: bool = False,
        annotate_size: int = 15,
        use_make_grid: bool = True,
        grid_column_size: int = 2,
        prompt2prompt_edit: bool=False,
        p2p_config: dict = None,
        use_inversion_attention: bool = True,
        source_prompt: str = None,
        traverse_p2p_config: bool = False,
        **args
    ) -> None:
        self.editing_prompts = editing_prompts
        self.clip_length = clip_length
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength

        if sample_seeds is None:
            max_num_samples_per_prompt = int(1e5)
            if num_samples_per_prompt > max_num_samples_per_prompt:
                raise ValueError
            sample_seeds = torch.randint(0, max_num_samples_per_prompt, (num_samples_per_prompt,))
            sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds

        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)

        self.annotate = annotate
        self.annotate_size = annotate_size
        self.make_grid = use_make_grid
        self.grid_column_size = grid_column_size
        self.prompt2prompt_edit = prompt2prompt_edit
        self.p2p_config = p2p_config
        self.use_inversion_attention = use_inversion_attention
        self.source_prompt = source_prompt
        self.traverse_p2p_config =traverse_p2p_config

    def log_sample_images(
        self, pipeline: DiffusionPipeline,
        device: torch.device, step: int,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: torch.FloatTensor = None,
        latents: torch.FloatTensor = None,
        mask:torch.FloatTensor = None,
        editing_type:str = "attribute",
        uncond_embeddings_list: List[torch.FloatTensor] = None,
        save_dir = None,
        duration = 100,
        fps = 10,
        use_interpolater = True
    ):
        torch.cuda.empty_cache()
        samples_all = []
        attention_all = []
        # handle input image
        if image is not None:
            input_pil_images = pipeline.numpy_to_pil(tensor_to_numpy(image))[0]
            if self.annotate :
                samples_all.append([
                            annotate_image(image, "input sequence", font_size=self.annotate_size) for image in input_pil_images
                        ])
            else:
                samples_all.append(input_pil_images)
        if isinstance(self.editing_prompts,str):
            self.editing_prompts = [self.editing_prompts]
        for idx, prompt in enumerate(tqdm(self.editing_prompts, desc="Generating sample images")):
            # if self.prompt2prompt_edit:
            #     if self.traverse_p2p_config:
            #         p2p_config_now = copy.deepcopy(self.p2p_config[idx])
            #     else:
            #         p2p_config_now = copy.deepcopy(self.p2p_config[idx])

            #     if idx == 0 and not self.use_inversion_attention:
            #         edit_type = 'save'
            #         p2p_config_now.update({'save_self_attention': True})
            #         print('Reflash the attention map in pipeline')

            #     else:
            #         edit_type = 'swap'
            #         p2p_config_now.update({'save_self_attention': False})

            #     p2p_config_now.update({'use_inversion_attention': self.use_inversion_attention})
            # else:
            #     edit_type = None

            input_prompt = prompt

            # generator = torch.Generator(device=device)
            # generator.manual_seed(seed)
            generator = None
            sequence = []
            window = 8
            window = min(window,self.clip_length)
            start_frame = 0
            end_frame = window
            patch_index = 0
            while start_frame < self.clip_length:
                torch.cuda.empty_cache()
                if patch_index == 0:
                    sequence_return = pipeline(
                        prompt=input_prompt,
                        source_prompt = self.editing_prompts[0] if self.source_prompt is None else self.source_prompt,
                        # edit_type = edit_type,
                        image=image[[0] + [0] + list(range(start_frame,min(self.clip_length,end_frame))),], # torch.Size([8, 3, 512, 512])
                        strength=self.strength,
                        generator=generator,
                        # window = 1,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=1,
                        # used in null inversion
                        editing_type = editing_type,
                        latents = [timestep_latent[:, :,[0] + [0] + list(range(start_frame,min(self.clip_length,end_frame))), :, :] for timestep_latent in latents],
                        mask = mask[:,:, [0] + [0] + list(range(start_frame, min(self.clip_length,end_frame))),] if mask is not None else None,
                        # latents = [timestep_latent[:, :,list(range(start_frame,min(self.clip_length,end_frame))), :, :] for timestep_latent in latents],
                        # mask = mask[:,:, list(range(start_frame, min(self.clip_length,end_frame))),] if mask is not None else None,
                        uncond_embeddings_list = uncond_embeddings_list,
                        save_path = save_dir,
                        # **p2p_config_now,
                    )
                else:
                    sequence_return = pipeline(
                        prompt=input_prompt,
                        reference_global_latents = reference_global_latents,
                        reference_latents = reference_latents,
                        source_prompt = self.editing_prompts[0] if self.source_prompt is None else self.source_prompt,
                        # edit_type = edit_type,
                        image=image[[0] + list(range(start_frame - 1,min(self.clip_length,end_frame))),], # torch.Size([8, 3, 512, 512])
                        strength=self.strength,
                        generator=generator,
                        # window = window,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=1,
                        # used in null inversion
                        editing_type = editing_type,
                        latents = [timestep_latent[:, :,[0] + list(range(start_frame-1,min(self.clip_length,end_frame))), :, :] for timestep_latent in latents],
                        mask = mask[:,:, [0] + list(range(start_frame-1, min(self.clip_length,end_frame))),] if mask is not None else None,
                        # latents = [timestep_latent[:, :,list(range(start_frame,min(self.clip_length,end_frame))), :, :] for timestep_latent in latents],
                        # mask = mask[:,:, list(range(start_frame, min(self.clip_length,end_frame))),] if mask is not None else None,
                        uncond_embeddings_list = uncond_embeddings_list,
                        save_path = save_dir,
                        # **p2p_config_now,
                    )
                start_frame = end_frame
                end_frame = end_frame + window
                if patch_index == 0:
                    reference_global_latents = sequence_return['reference_global_latents']
                reference_latents = sequence_return['reference_latents']
                patch_index = patch_index + 1
                # if self.prompt2prompt_edit:
                #     sequence_temp = sequence_return['sdimage_output'].images[0]
                #     # attention_output = sequence_return['attention_output']
                # else:
                #     sequence_temp = sequence_return.images[0]
                sequence_temp = sequence_return['sdimage_output'].images[0]
                sequence = sequence + sequence_temp
                torch.cuda.empty_cache()
            # sequence = torch.cat(sequence,dim = 2)

            if self.annotate:
                images = [
                    annotate_image(image, prompt, font_size=self.annotate_size) for image in sequence
                ]
            else:
                images = sequence
            control_images = []
            for i in range(control_image.shape[2]):
                control_images.append(Image.fromarray((control_image[0,:,i]*255).cpu().numpy().transpose(1,2,0).astype(np.uint8)))
            #smoother start
            if use_interpolater:
                for i in range(len(images)):
                    images[i] = np.array(images[i]).transpose(2,0,1)[None:]/255
                frames = torch.from_numpy(np.stack(images, axis= 0)).cuda()
                f, C, H, W = frames.shape
                ph = ((H - 1) // 32 + 1) * 32
                pw = ((W - 1) // 32 + 1) * 32
                padding = (0, pw - W, 0, ph - H)
                frames = F.pad(frames,padding)
                smoother = Model()
                smoother.load_model('RIFEModel', -1)
                print('using smoother')
                with torch.no_grad():
                    for i in range(f - 2):
                        img0 = frames[i:i+1].float()
                        img1 = frames[i+2:i+3].float()
                        mid = smoother.inference(img0,img1)
                        mid_padded = F.pad(mid,padding)
                        frames[i+1:i+2,] = (frames[i+1:i+2,] + mid_padded[None:])/2
                        torch.cuda.empty_cache()
                images = []
                for i in range(len(frames)):
                    images.append(Image.fromarray((frames[i] * 255).cpu().numpy().astype(np.uint8).transpose(1,2,0)))
            # smoother end
            if self.make_grid:
                samples_all.append(control_images)
                samples_all.append(images)
                # if self.prompt2prompt_edit:
                #     if attention_output is not None:
                #         attention_all.append(attention_output)

            save_path = os.path.join(self.logdir, f"step_{step}_{idx}.gif")
            save_gif_mp4_folder_type(images, save_path,duration = duration,fps = fps)

            # if self.prompt2prompt_edit:

            #     if attention_output is not None:
            #         save_gif_mp4_folder_type(attention_output, save_path.replace('.gif', 'atten.gif'),duration = duration,fps = fps)

        if self.make_grid:
            samples_all = [make_grid(images, cols=int(len(samples_all))) for images in zip(*samples_all)]
            save_path = os.path.join(self.logdir, f"step_{step}.gif")
            save_gif_mp4_folder_type(samples_all, save_path,duration = duration,fps = fps)
            if self.prompt2prompt_edit:
                if len(attention_all) > 0 :
                    attention_all = [make_grid(images, cols=1) for images in zip(*attention_all)]
                if len(attention_all) > 0:
                    save_gif_mp4_folder_type(attention_all, save_path.replace('.gif', 'atten.gif'),duration = duration,fps = fps)
        return samples_all




def tensor_to_numpy(image, b=1):
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

    image = image.cpu().float().numpy()
    image = rearrange(image, "(b f) c h w -> b f h w c", b=b)
    return image
