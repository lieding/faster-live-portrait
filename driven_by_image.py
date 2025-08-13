
import shutil
from typing import Optional, Dict, Any
import io
import os
import subprocess
# import uvicorn
import cv2
import time
import numpy as np
import os
import datetime
import platform
import pickle

from omegaconf import OmegaConf
from pydantic import BaseModel

from fastapi.responses import StreamingResponse
from zipfile import ZipFile

from fastapi import APIRouter, Depends, FastAPI, Request, Response, UploadFile
from fastapi import File, Body, Form

from src.pipelines.faster_live_portrait_pipeline_customized import FasterLivePortraitPipeline
from src.utils import logger

global pipe

def check_all_checkpoints_exist(infer_cfg):
    """
    check whether all checkpoints exist
    :return:
    """
    ret = True
    for name in infer_cfg.models:
        if not isinstance(infer_cfg.models[name].model_path, str):
            for i in range(len(infer_cfg.models[name].model_path)):
                infer_cfg.models[name].model_path[i] = infer_cfg.models[name].model_path[i].replace("./checkpoints",
                                                                                                    checkpoints_dir)
                if not os.path.exists(infer_cfg.models[name].model_path[i]) and not os.path.exists(
                        infer_cfg.models[name].model_path[i][:-4] + ".onnx"):
                    return False
        else:
            infer_cfg.models[name].model_path = infer_cfg.models[name].model_path.replace("./checkpoints",
                                                                                          checkpoints_dir)
            if not os.path.exists(infer_cfg.models[name].model_path) and not os.path.exists(
                    infer_cfg.models[name].model_path[:-4] + ".onnx"):
                return False
    for name in infer_cfg.animal_models:
        if not isinstance(infer_cfg.animal_models[name].model_path, str):
            for i in range(len(infer_cfg.animal_models[name].model_path)):
                infer_cfg.animal_models[name].model_path[i] = infer_cfg.animal_models[name].model_path[i].replace(
                    "./checkpoints",
                    checkpoints_dir)
                if not os.path.exists(infer_cfg.animal_models[name].model_path[i]) and not os.path.exists(
                        infer_cfg.animal_models[name].model_path[i][:-4] + ".onnx"):
                    return False
        else:
            infer_cfg.animal_models[name].model_path = infer_cfg.animal_models[name].model_path.replace("./checkpoints",
                                                                                                        checkpoints_dir)
            if not os.path.exists(infer_cfg.animal_models[name].model_path) and not os.path.exists(
                    infer_cfg.animal_models[name].model_path[:-4] + ".onnx"):
                return False

    # XPOSE
    xpose_model_path = os.path.join(checkpoints_dir, "liveportrait_animal_onnx/xpose.pth")
    if not os.path.exists(xpose_model_path):
        return False
    embeddings_cache_9_path = os.path.join(checkpoints_dir, "liveportrait_animal_onnx/clip_embedding_9.pkl")
    if not os.path.exists(embeddings_cache_9_path):
        return False
    embeddings_cache_68_path = os.path.join(checkpoints_dir, "liveportrait_animal_onnx/clip_embedding_68.pkl")
    if not os.path.exists(embeddings_cache_68_path):
        return False
    return ret



def convert_onnx_to_trt_models(infer_cfg):
    ret = True
    for name in infer_cfg.models:
        if not isinstance(infer_cfg.models[name].model_path, str):
            for i in range(len(infer_cfg.models[name].model_path)):
                trt_path = infer_cfg.models[name].model_path[i]
                onnx_path = trt_path[:-4] + ".onnx"
                if not os.path.exists(trt_path):
                    convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                    logger_f.info(f"convert onnx model: {onnx_path}")
                    result = subprocess.run(convert_cmd, shell=True, check=True)
                    # 检查结果
                    if result.returncode == 0:
                        logger_f.info(f"convert onnx model: {onnx_path} successful")
                    else:
                        logger_f.error(f"convert onnx model: {onnx_path} failed")
                        return False
        else:
            trt_path = infer_cfg.models[name].model_path
            onnx_path = trt_path[:-4] + ".onnx"
            if not os.path.exists(trt_path):
                convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                logger_f.info(f"convert onnx model: {onnx_path}")
                result = subprocess.run(convert_cmd, shell=True, check=True)
                # 检查结果
                if result.returncode == 0:
                    logger_f.info(f"convert onnx model: {onnx_path} successful")
                else:
                    logger_f.error(f"convert onnx model: {onnx_path} failed")
                    return False

    for name in infer_cfg.animal_models:
        if not isinstance(infer_cfg.animal_models[name].model_path, str):
            for i in range(len(infer_cfg.animal_models[name].model_path)):
                trt_path = infer_cfg.animal_models[name].model_path[i]
                onnx_path = trt_path[:-4] + ".onnx"
                if not os.path.exists(trt_path):
                    convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                    logger_f.info(f"convert onnx model: {onnx_path}")
                    result = subprocess.run(convert_cmd, shell=True, check=True)
                    # 检查结果
                    if result.returncode == 0:
                        logger_f.info(f"convert onnx model: {onnx_path} successful")
                    else:
                        logger_f.error(f"convert onnx model: {onnx_path} failed")
                        return False
        else:
            trt_path = infer_cfg.animal_models[name].model_path
            onnx_path = trt_path[:-4] + ".onnx"
            if not os.path.exists(trt_path):
                convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                logger_f.info(f"convert onnx model: {onnx_path}")
                result = subprocess.run(convert_cmd, shell=True, check=True)
                # 检查结果
                if result.returncode == 0:
                    logger_f.info(f"convert onnx model: {onnx_path} successful")
                else:
                    logger_f.error(f"convert onnx model: {onnx_path} failed")
                    return False
    return ret


class LivePortraitParams(BaseModel):
    flag_pickle: bool = False
    flag_relative_input: bool = True
    flag_do_crop_input: bool = True
    flag_remap_input: bool = True
    driving_multiplier: float = 1.0
    flag_stitching: bool = True
    flag_crop_driving_video_input: bool = True
    flag_video_editing_head_rotation: bool = False
    flag_is_animal: bool = False
    scale: float = 2.3
    vx_ratio: float = 0.0
    vy_ratio: float = -0.125
    scale_crop_driving_video: float = 2.2
    vx_ratio_crop_driving_video: float = 0.0
    vy_ratio_crop_driving_video: float = -0.1
    driving_smooth_observation_variance: float = 1e-7

async def startup_event():
    global pipe
    # default use trt model
    cfg_file = os.path.join(project_dir, "configs/trt_infer.yaml")
    infer_cfg = OmegaConf.load(cfg_file)
    checkpoints_exist = check_all_checkpoints_exist(infer_cfg)

    # first: download model if not exist
    if not checkpoints_exist:
        download_cmd = f"huggingface-cli download warmshao/FasterLivePortrait --local-dir {checkpoints_dir}"
        logger_f.info(f"download model: {download_cmd}")
        result = subprocess.run(download_cmd, shell=True, check=True)
        # 检查结果
        if result.returncode == 0:
            logger_f.info(f"Download checkpoints to {checkpoints_dir} successful")
        else:
            logger_f.error(f"Download checkpoints to {checkpoints_dir} failed")
            exit(1)
    # second: convert onnx model to trt
    convert_ret = convert_onnx_to_trt_models(infer_cfg)
    if not convert_ret:
        logger_f.error(f"convert onnx model to trt failed")
        exit(1)

    infer_cfg.infer_params.flag_pasteback = True
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=True)


# model dir
project_dir = os.path.dirname(__file__)
checkpoints_dir = os.environ.get("FLIP_CHECKPOINT_DIR", os.path.join(project_dir, "checkpoints"))
log_dir = os.path.join(project_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
result_dir = os.path.join(project_dir, "results")
os.makedirs(result_dir, exist_ok=True)

logger_f = logger.get_logger("faster_liveportrait_api", log_file=os.path.join(log_dir, "log_run.log"))


def run_with_image(source_image: np.ndarray, driving_image: np.ndarray, save_dir):
    """
    Run expression editing with a single driving image instead of video
    """
    global pipe
    ret = pipe.prepare_source(source_image, realtime=False)
    if not ret:
        logger_f.warning(f"no face in source image! exit!")
        return

    # Load driving image
    driving_img_bgr = driving_image #cv2.imread(driving_image_path, cv2.IMREAD_COLOR)
    if driving_img_bgr is None:
        logger_f.warning(f"Could not load driving image")
        return
    
    h, w = pipe.src_imgs[0].shape[:2]
    
    # Process the single driving image
    t0 = time.time()
    dri_crop, out_crop, out_org, dri_motion_info = pipe.run(
        driving_img_bgr, 
        pipe.src_imgs[0], 
        pipe.src_infos[0],
        first_frame=True  # Always treat as first frame for single image
    )
    
    if out_crop is None:
        logger_f.warning(f"no face in driving image")
        return

    infer_time = time.time() - t0
    logger_f.info(f"inference time: {infer_time * 1000:.2f} ms")
    
    # Save results
    result_filename = f"output"
    
    # Save crop version (side by side comparison)
    if dri_crop is not None:
        dri_crop_resized = cv2.resize(dri_crop, (512, 512))
        out_crop_resized = cv2.resize(out_crop, (512, 512))
        comparison_crop = np.concatenate([dri_crop_resized, out_crop_resized], axis=1)
        comparison_crop_bgr = cv2.cvtColor(comparison_crop, cv2.COLOR_RGB2BGR)
        crop_save_path = os.path.join(save_dir, f"{result_filename}-crop-comparison.jpg")
        cv2.imwrite(crop_save_path, comparison_crop_bgr)
        logger_f.info(f"Saved crop comparison: {crop_save_path}")
    
    # Save original size result
    out_org_bgr = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
    org_save_path = os.path.join(save_dir, f"{result_filename}-result.jpg")
    cv2.imwrite(org_save_path, out_org_bgr)
    logger_f.info(f"Saved result: {org_save_path}")
    
    # Save individual outputs
    out_crop_bgr = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
    crop_only_save_path = os.path.join(save_dir, f"{result_filename}-crop-result.jpg")
    cv2.imwrite(crop_only_save_path, out_crop_bgr)
    
    # Save driving motion as pkl for potential reuse
    motion_lst = [dri_motion_info[0]] # type: ignore
    c_eyes_lst = [dri_motion_info[1]] # type: ignore
    c_lip_lst = [dri_motion_info[2]] # type: ignore
    
    template_dct = {
        'n_frames': 1,
        'output_fps': 1,  # Not relevant for single image
        'motion': motion_lst,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }
    template_pkl_path = os.path.join(save_dir, f"{result_filename}-motion.pkl")
    with open(template_pkl_path, "wb") as fw:
        pickle.dump(template_dct, fw)
    logger_f.info(f"Saved motion data: {template_pkl_path}")


async def upload_single_image_files(
    source_image: np.ndarray,
    driving_image: np.ndarray,
    flag_is_animal: bool = False,
    flag_relative_input: bool = True,
    flag_do_crop_input: bool = True,
    flag_remap_input: bool = True,
    driving_multiplier: float = 1.0,
    flag_stitching: bool = True,
    flag_crop_driving_video_input: bool = True,
    flag_video_editing_head_rotation: bool = False,
    scale: float = 2.3,
    vx_ratio: float = 0.0,
    vy_ratio: float = -0.125,
    scale_crop_driving_video: float = 2.2,
    vx_ratio_crop_driving_video: float = 0.0,
    vy_ratio_crop_driving_video: float = -0.1,
    driving_smooth_observation_variance: float = 1e-7
):
    """
    Single image expression editing endpoint
    """
    if not source_image:
        return {"error": "Source image is required"}
    
    if not driving_image:
        return {"error": "Driving image is required"}
    
    # Create infer_params from form data
    infer_params = LivePortraitParams(
        flag_is_animal=flag_is_animal,
        flag_pickle=False,  # Always False for single image
        flag_relative_input=flag_relative_input,
        flag_do_crop_input=flag_do_crop_input,
        flag_remap_input=flag_remap_input,
        driving_multiplier=driving_multiplier,
        flag_stitching=flag_stitching,
        flag_crop_driving_video_input=flag_crop_driving_video_input,
        flag_video_editing_head_rotation=flag_video_editing_head_rotation,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        scale_crop_driving_video=scale_crop_driving_video,
        vx_ratio_crop_driving_video=vx_ratio_crop_driving_video,
        vy_ratio_crop_driving_video=vy_ratio_crop_driving_video,
        driving_smooth_observation_variance=driving_smooth_observation_variance
    )

    global pipe
    pipe.init_vars()
    if infer_params.flag_is_animal != pipe.is_animal:
        pipe.init_models(is_animal=infer_params.flag_is_animal)

    # Update pipeline configuration
    args_user = {
        'flag_relative_motion': infer_params.flag_relative_input,
        'flag_do_crop': infer_params.flag_do_crop_input,
        'flag_pasteback': infer_params.flag_remap_input,
        'driving_multiplier': infer_params.driving_multiplier,
        'flag_stitching': infer_params.flag_stitching,
        'flag_crop_driving_video': infer_params.flag_crop_driving_video_input,
        'flag_video_editing_head_rotation': infer_params.flag_video_editing_head_rotation,
        'src_scale': infer_params.scale,
        'src_vx_ratio': infer_params.vx_ratio,
        'src_vy_ratio': infer_params.vy_ratio,
        'dri_scale': infer_params.scale_crop_driving_video,
        'dri_vx_ratio': infer_params.vx_ratio_crop_driving_video,
        'dri_vy_ratio': infer_params.vy_ratio_crop_driving_video,
    }
    pipe.update_cfg(args_user)

    # Create temporary directory for uploaded files
    temp_dir = os.path.join(result_dir, f"temp-single-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
    os.makedirs(temp_dir, exist_ok=True)

    try:

        # Create results directory
        save_dir = os.path.join(result_dir, f"single-image-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)

        # Process the images
        run_with_image(source_image, driving_image, save_dir)

        # Create zip file with results
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as zip_file:
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, arcname=os.path.relpath(file_path, save_dir))

        zip_buffer.seek(0)
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        shutil.rmtree(save_dir)

        # Return zip file
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=single_image_results.zip"}
        )

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if 'save_dir' in locals() and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        
        logger_f.error(f"Error processing single image request: {str(e)}")
        return {"error": f"Processing failed: {str(e)}"}



if __name__ == "__main__":
    source_path = ""
    driving_image_path = ""
    source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    driving = cv2.imread(driving_image_path, cv2.IMREAD_COLOR)
    upload_single_image_files(source_image=source_img, driving_image=driving) # type: ignore