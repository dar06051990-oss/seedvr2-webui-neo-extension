# seedvr2-webui-neo-extension
 ## Forge Neo compatibility fixes

This fork adds several fixes for Forge Neo:

- fixes the 💾 save/download button after SeedVR2 upscaling;
- preserves Seed and generation infotext correctly;
- adds an optional **Auto-save SeedVR2 Result** setting;
- auto-save is disabled by default, so results are not duplicated to the output folder unless enabled.

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
*   **img2img**: Bypasses the standard img2img processing and directly applies SeedVR2 upscaling to the input image.

<img width="1660" height="694" alt="image" src="https://github.com/user-attachments/assets/777c34e7-aca6-4e51-9994-f02f817311ea" />

