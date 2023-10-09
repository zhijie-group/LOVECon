# code mostly taken from https://github.com/huggingface/diffusers

from typing import Callable, List, Optional, Union,Dict,Any
import os, sys
import PIL
import torch
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm
import cv2
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from torch.nn import functional as F
from ..models.unet_3d_condition import UNetPseudo3DConditionModel
from ..models.controlnet_3d_condition import ControlNetPseudo3DModel,MultiControlNetPseudo3DModel
from .stable_diffusion_controlnet import SpatioTemporalStableDiffusionControlnetPipeline
from video_diffusion.prompt_attention import attention_util
from video_diffusion.common.image_util import save_gif_mp4_folder_type
from annotator.util import HWC3,get_control
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange
import imageio
import shutil

import kornia
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class P2pDDIMSpatioTemporalControlnetPipeline(SpatioTemporalStableDiffusionControlnetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        controlnet:ControlNetPseudo3DModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,],
        control_type:str,
        editing_type:str,
        dilation_kernel = None,
        disk_store: bool=False
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet,scheduler,control_type)
        self.editing_type = editing_type
        self.dilation_kernel = dilation_kernel 
        self.store_controller = attention_util.AttentionStore(disk_store=disk_store)
        self.empty_controller = attention_util.EmptyControl()
    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """
    def add_new_scheduler(self, new_scheduler):
        self.new_scheduler = new_scheduler
    
    def check_inputs(self, prompt, height, width, callback_steps,strength=None):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if strength is not None:
            if strength <= 0 or strength > 1:
                raise ValueError(f"The value of strength should in (0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )


    @torch.no_grad()
    def prepare_latents_ddim_inverted(self, image, batch_size, num_images_per_prompt, 
                                        text_embeddings,controls,
                                        edit_phrase,
                                        store_attention=False, prompt=None,
                                        controlnet_conditioning_scale=1.0,
                                        generator=None,
                                        LOW_RESOURCE = True,
                                        save_path = None
                                      ):
        self.prepare_before_train_loop()
        window = 8
        if edit_phrase == '':
            store_attention = False
        if store_attention:
            attention_util.register_attention_control(self, self.store_controller)
        resource_default_value = self.store_controller.LOW_RESOURCE
        self.store_controller.LOW_RESOURCE = LOW_RESOURCE  # in inversion, no CFG, record all latents attention
        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            # print('image',image.shape)
            # init_latents = self.vae.encode(image).latent_dist.sample(generator)
            init_latents = []
            start_frame = 0
            end_frame = window
            while start_frame < image.shape[0]:
                init_latents_temp = self.vae.encode(image[start_frame:end_frame]).latent_dist.sample(generator)
                init_latents.append(init_latents_temp)
                start_frame = end_frame
                end_frame = end_frame + window
                torch.cuda.empty_cache()
            init_latents = torch.cat(init_latents, dim = 0)

        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        # get latents
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)
        ddim_latents_all_step_list = []
        mask_list = []
        attention_list = []
        start_frame = 0
        end_frame = window
        mask = None
        while start_frame < init_latents_bcfhw.shape[2]:
            self.store_controller.reset()
            ddim_latents_all_step = self.ddim_clean2noisy_loop(init_latents_bcfhw[:,:,start_frame:end_frame], text_embeddings, 
                                                            controls[:,:,start_frame:end_frame],controlnet_conditioning_scale,self.store_controller)
            if edit_phrase is not None and edit_phrase != "":
                mask = attention_util.get_attention_mask(self.tokenizer, prompt, edit_phrase,
                                                        self.store_controller, 16, ["up", "down"],editing_type = self.editing_type,dilation_kernel = self.dilation_kernel)
                mask_list.append(mask)
            ddim_latents_all_step_list.append(ddim_latents_all_step)
            if store_attention and (save_path is not None) :
                os.makedirs(save_path+'/mask',exist_ok = True)
                attention_output = attention_util.show_cross_attention(self.tokenizer, prompt, 
                                                                    self.store_controller, 16, ["up", "down"],
                                                                    save_path = save_path+'/mask', editing_type = self.editing_type, dilation_kernel = self.dilation_kernel)
                attention_list = attention_list + attention_output
            start_frame = end_frame
            end_frame = end_frame + window
            torch.cuda.empty_cache()
            # Detach the controller for safety
        if edit_phrase is not None and edit_phrase != "":
            mask = torch.cat(mask_list,dim = 2)
            video_save_path = f'{save_path}/mask.gif'
            save_gif_mp4_folder_type(attention_list,video_save_path)
        ddim_latents_all_step = [torch.cat(sublist, dim=2) for sublist in zip(*ddim_latents_all_step_list)]
        attention_util.register_attention_control(self, self.empty_controller)
        self.store_controller.LOW_RESOURCE = resource_default_value
        return ddim_latents_all_step, mask
    
    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings,control_image, controlnet_conditioning_scale=1.0,controller=None):
        weight_dtype = latent.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        print('Invert clean image to noise latents by DDIM and Unet and controlnet')
        for i in trange(len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            # [1, 4, 8, 64, 64] ->  [1, 4, 8, 64, 64])
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent,
                t,
                encoder_hidden_states=cond_embeddings,
                controlnet_cond=control_image,
                return_dict=False,
            )
            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= controlnet_conditioning_scale
            noise_pred = self.unet(
                latent,
                t,
                encoder_hidden_states=cond_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )["sample"]

            # noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            
            latent = self.next_clean2noise_step(noise_pred, t, latent)
            if controller is not None: controller.step_callback(latent)
            all_latent.append(latent.to(dtype=weight_dtype))
        return all_latent
    
    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """
        Assume the eta in DDIM=0
        """
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    def p2preplace_edit(self, **kwargs):
        # Edit controller during inference
        # The controller must know the source prompt for replace mapping
        
        len_source = {len(kwargs['source_prompt'].split(' '))}
        len_target = {len(kwargs['prompt'].split(' '))}
        equal_length = (len_source == len_target)
        print(f" len_source: {len_source}, len_target: {len_target}, equal_length: {equal_length}")
        # edit_controller = attention_util.make_controller(
        #                     self.tokenizer, 
        #                     [ kwargs['source_prompt'], kwargs['prompt']],
        #                     NUM_DDIM_STEPS = kwargs['num_inference_steps'],
        #                     is_replace_controller=kwargs.get('is_replace_controller', True) and equal_length,
        #                     cross_replace_steps=kwargs['cross_replace_steps'], 
        #                     self_replace_steps=kwargs['self_replace_steps'], 
        #                     blend_words=kwargs.get('blend_words', None),
        #                     equilizer_params=kwargs.get('eq_params', None),
        #                     additional_attention_store=self.store_controller,
        #                     use_inversion_attention = kwargs['use_inversion_attention'],
        #                     blend_th = kwargs.get('blend_th', (0.3, 0.3)),
        #                     blend_self_attention = kwargs.get('blend_self_attention', None),
        #                     blend_latents=kwargs.get('blend_latents', None),
        #                     save_path=kwargs.get('save_path', None),
        #                     save_self_attention = kwargs.get('save_self_attention', True),
        #                     disk_store = kwargs.get('disk_store', False)
        #                     )
        
        # attention_util.register_attention_control(self, edit_controller)
        # attention_util.register_attention_control(self, self.empty_controller)
        # In ddim inferece, no need source prompt
        reference_global_latents,reference_latents,sdimage_output = self.sd_ddim_pipeline(
            # controller = edit_controller, 
            # controller = self.empty_controller,
            # controller = self.store_controller,
            # target_prompt = kwargs['prompts'][1],
            **kwargs)
        # if hasattr(edit_controller.latent_blend, 'mask_list'):
        #     mask_list = edit_controller.latent_blend.mask_list
        # else:
        #     mask_list = None
        # if len(edit_controller.attention_store.keys()) > 0:
        #     attention_output = attention_util.show_cross_attention(self.tokenizer, kwargs['prompt'], 
        #                                                        edit_controller, 16, ["up", "down"])
        # else:
        #     attention_output = None
        attention_output = None
        mask_list = None
        dict_output = {
                "sdimage_output" : sdimage_output,
                "attention_output" : attention_output,
                "mask_list" : mask_list,
                "reference_global_latents": reference_global_latents,
                "reference_latents" : reference_latents,
            }
        # attention_util.register_attention_control(self, self.empty_controller)
        return dict_output
    
    def latents_fusion(self,i,editing_type,latents,source_latents,mask = None):
        if editing_type == "attribute":
            if i < 30:
                latents = latents * 0.97 + source_latents * 0.03
            if mask is not None and i < 40:
                latents = latents * mask + source_latents * (1 - mask)
        elif editing_type == "background":
            if mask is not None and i < 20:
                latents = latents * (1 - mask) + source_latents * mask
        elif editing_type == "shape":
            if mask is not None and i < 40:
                latents = latents * mask + source_latents * (1 - mask)
            # else:
            #     latents = latents * 0.95 + source_latents * 0.05
            pass
        elif editing_type == "style":
            if i < 20:
                latents = latents * 0.95 + source_latents * 0.05
            # if i > 40:
            #     latents = latents * 0.96 + source_latents * 0.04
        return latents

    @torch.no_grad()
    def get_control_image(self,images,apply_control):
        control = []
        if len(images.shape) == 4:
            images = rearrange(images,'f c h w -> f h w c')
        elif len(images.shape) == 5:
            images = rearrange(images.squeeze(0),'c f h w -> f h w c')
        # compute control for each frame
        for i in images:
            i = (i + 1) /2 * 255
            if self.control_type == 'canny':
                detected_map = apply_control(i.cpu().numpy().astype(np.uint8), 100, 100)
                # canny_image = detected_map[:, :, None]
                # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                # canny_image = Image.fromarray(canny_image)
                # path = './canny'
                # i = len(os.listdir(path))
                # imageio.imsave(os.path.join('./canny',f'canny{i:02}.jpg'),canny_image)

            elif self.control_type == 'openpose' or self.control_type == 'depth':
                detected_map, _ = apply_control(i.cpu().numpy().astype(np.uint8))
            elif self.control_type == 'hed' or self.control_type == 'seg':
                detected_map = apply_control(i.cpu().numpy().astype(np.uint8))
            elif self.control_type == 'scribble':
                i = i.cpu().numpy().astype(np.uint8)
                detected_map = np.zeros_like(i, dtype=np.uint8)
                detected_map[np.min(i, axis=2) < control_config.value] = 255
            elif self.control_type == 'normal':
                _, detected_map = apply_control(i.cpu().numpy().astype(np.uint8), bg_th=control_config.bg_threshold)
            elif self.control_type == 'mlsd':
                detected_map = apply_control(i.cpu().numpy().astype(np.uint8), control_config.value_threshold, control_config.distance_threshold)
            else:
                raise ValueError(self.control_type)
            control.append(HWC3(detected_map))

        # stack control with all frames with shape [b c f h w]
        control = np.stack(control)
        control = np.array(control).astype(np.float32) / 255.0
        control = torch.from_numpy(control)
        control = control.unsqueeze(0) #[f h w c] -> [b f h w c ]
        control = rearrange(control, "b f h w c -> b c f h w")
        return control.to(images.device)



    @torch.no_grad()
    def __call__(self, **kwargs):
                
        reference_global_latents, reference_latents, sdimage_output = self.sd_ddim_pipeline(controller = self.store_controller, **kwargs)
        dict_output = {
            "sdimage_output" : sdimage_output,
            "reference_global_latents": reference_global_latents,
            'reference_latents': reference_latents
        }
        return dict_output

    
    @torch.no_grad()
    def sd_ddim_pipeline(
        self,
        prompt: Union[str, List[str]],
        reference_global_latents = None,
        reference_latents = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        # window:Optional[int] = 8,
        strength: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        editing_type:Optional[str] = "attribute",
        latents: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        new_scheduler = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controller: attention_util.AttentionControl = None,
        mask:Optional[torch.FloatTensor] = None,
        **args
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Only used in DDIM or strength<1.0
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.            
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # 1. Check inputs. Raise error if not correct
        print("guidance_scale:",guidance_scale)
        print("condition scale:",controlnet_conditioning_scale)
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            strength
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 4.1 Prepare control image
        apply_control = get_control(self.control_type)

        control_image = self.get_control_image(
            images=image,
            apply_control = apply_control
        )

        # 5. Prepare latent variables
        
        if latents is None:
            ddim_latents_all_step = self.prepare_latents_ddim_inverted(
                image, batch_size, num_images_per_prompt, 
                text_embeddings,control_image,
                store_attention=False, # avoid recording attention in first inversion
                generator = generator,
            )
            latents = ddim_latents_all_step[-1]
        else:
            ddim_latents_all_step=None
        
        latents_dtype = latents[-1].dtype

            
        All_latents = latents

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop

        latents = latents[-1]
        if mask is not None:
            mask = mask.to(latents_dtype).to(self._execution_device)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        reference_global_latents_new = None
        if reference_global_latents is None :
            reference_global_latents_new = [latents[:,:,0:1]]
        reference_latents_new = [latents[:,:,-1:]]
        # reference_all_latents_new2 = [latents[:,:,1:2]]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(timesteps)):
                torch.cuda.empty_cache()
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=control_image,
                    return_dict=False,
                )
                down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= controlnet_conditioning_scale

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0].to(dtype=latents_dtype)
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.latents_fusion(i,editing_type,latents,All_latents[-i-1],mask = mask)

                # compute the previous noisy sample x_t -> x_t-1
                if hasattr(self,'new_scheduler') and i > 48 and latents.shape[2] >=3:
                    latents = self.new_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if reference_global_latents is None:
                    reference_global_latents_new.append(latents[:,:,0:1])
                reference_latents_new.append(latents[:,:,-1:])

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                torch.cuda.empty_cache()

        # 8. Post-processing
        image = self.decode_latents(latents[:,:,2:])

        # 9. Run safety checker
        has_nsfw_concept = None

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()
            
        return reference_global_latents_new, reference_latents_new, StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def print_pipeline(self, logger):
        print('Overview function of pipeline: ')
        print(self.__class__)

        print(self)
        
        expected_modules, optional_parameters = self._get_signature_keys(self)        
        components_details = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }
        import json
        logger.info(str(components_details))
        # logger.info(str(json.dumps(components_details, indent = 4)))
        # print(str(components_details))
        # print(self._optional_components)
        
        print(f"python version {sys.version}")
        print(f"torch version {torch.__version__}")
        print(f"validate gpu status:")
        print( torch.tensor(1.0).cuda()*2)
        os.system("nvcc --version")

        import diffusers
        print(diffusers.__version__)
        print(diffusers.__file__)

        try:
            import bitsandbytes
            print(bitsandbytes.__file__)
        except:
            print("fail to import bitsandbytes")
        # os.system("accelerate env")
        # os.system("python -m xformers.info")
