# seedvr2-webui-neo-extension
 ## Forge Neo compatibility fixes and additions

This fork keeps the original SeedVR2 WebUI extension behavior and adds Forge Neo compatibility fixes plus extra model support:

- fixes the 💾 save/download button after SeedVR2 upscaling;
- preserves Seed and generation infotext correctly;
- adds an optional **Auto-save SeedVR2 Result** setting;
- auto-save is disabled by default, so results are not duplicated to the output folder unless enabled;
- adds an **Img2Img Mode** selector:
  - **SeedVR2 only (upscale input image)** — original behavior;
  - **Run Forge img2img, then SeedVR2** — runs the normal Forge img2img pipeline first and then upscales its result;
- adds native loading for Forge/Comfy quantized `.safetensors` SeedVR2 models using `*.comfy_quant` metadata:
  - **INT8 Tensorwise / ConvRot** (`int8_tensorwise`);
  - **W4A8 ConvRot** (`asym_w4a8_int8`);
  - **INT4 / W4A4 ConvRot** (`convrot_w4a4`);
  - mixed INT4 checkpoints with INT8 fallback layers are supported;
- quantized Linear weights remain packed in memory and execute through `comfy-kitchen` instead of being expanded to BF16 during loading.

Regular FP16/BF16/FP8/GGUF loading paths remain available.

Native quantized `.safetensors` support requires a Forge Neo build with `comfy-kitchen>=0.2.15` available.

Based on the original project by yamosin.

---

如果 `install.py`没有正确运行，请自己在环境里使用以下命令安装
```
pip install rotary-embedding-torch
```

seedvr2模型请放置在`./model/seedvr2`文件夹下，包括`ema_vae_fp16.safetensors`和seedvr2模型如`seedvr2_ema_7b_sharp-Q4_K_M.gguf`（或safetensors）



If `install.py` fails to execute properly, please manually run the following command in your environment:

```
pip install rotary-embedding-torch
```

Please place the SeedVR2 models in the `./model/seedvr2` directory. This includes `ema_vae_fp16.safetensors` and SeedVR2 models such as `seedvr2_ema_7b_sharp-Q4_K_M.gguf` (or `.safetensors` files).

## 使用
通过最下面的 脚本 使用此扩展，一般仅需要修改 Upscale Resolution (Shortest Edge)以提高分辨率，该脚本在txt2img和img2img有不同运作方式：

txt2img:在所有扩展和图像生成之后，获取图像并进行seedvr2上采样

img2img:跳过img2img，直接使用seedvr2上采样


Use this extension via the **Script** dropdown menu at the bottom of the page. Generally, you only need to adjust the **Upscale Resolution (Shortest Edge)** to increase the resolution.

The script functions differently depending on the mode:

*   **txt2img**: Performs SeedVR2 upscaling on the image *after* the generation process and all other extensions have completed.
*   **img2img / SeedVR2 only**: Bypasses the standard img2img processing and directly applies SeedVR2 upscaling to the input image.
*   **img2img / Run Forge img2img, then SeedVR2**: Runs the regular Forge img2img pipeline first, then applies SeedVR2 to the generated image.

<img width="1660" height="694" alt="image" src="https://github.com/user-attachments/assets/777c34e7-aca6-4e51-9994-f02f817311ea" />

