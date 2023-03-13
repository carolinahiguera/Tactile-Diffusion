import os
from os import path as osp
import hydra
import trimesh
from omegaconf import DictConfig
from render.digit_render import digit_renderer
from modules.pose import  extract_poses_real
from modules.mesh import sample_closest_point
from modules.misc import *
from viz.visualizer import Viz

def touch_simulator(cfg: DictConfig):
    """Tactile simulator function"""
    render_cfg = cfg.render
    obj_model = cfg.obj_model
    dataset_number = cfg.dataset_number
    randomize = render_cfg.randomize

    dataset_name = f"dataset_{dataset_number}"
    data_path = osp.join(DIRS["data"], obj_model, dataset_name)
    obj_path = osp.join(DIRS["obj_models"], obj_model, "nontextured.stl")

    device = get_device(cpu=True)
    gt_p_cam, gt_p, idx_removed = extract_poses_real(
        pose_file=osp.join(data_path, "synced_data.npy"),
        alignment_file=osp.join(data_path, "..", "alignment.npy"),
        obj_model=obj_model,
        device=device,
        subsample=1,
    )  # poses : (N , 4, 4)
    gt_p_cam = gt_p_cam.cpu().numpy()
    gt_p = gt_p.cpu().numpy()

    mesh = trimesh.load(obj_path)
    gt_p2 = sample_closest_point(mesh, gt_p)
    print(f"Getting depth images for {obj_model}, dataset {dataset_number} ...")

    # start renderer
    if randomize:
        tac_render = digit_renderer(cfg=render_cfg, obj_path=obj_path, randomize=randomize)
    else:
        tac_render = digit_renderer(cfg=render_cfg, obj_path=obj_path)
    
    (
        hm,
        cm,
        image,
        _,
        _,
        _,
    ) = tac_render.render_sensor_trajectory(p=gt_p2, mNoise=cfg.noise)
    
    print(f"Writing files for {obj_model}, dataset {dataset_number} ...")
    heightmap_path = osp.join(data_path, "frames_depth")
    os.makedirs(heightmap_path)
    save_images_heightmaps(heightmapsImages=hm, save_path=heightmap_path)
    np.save(f"{data_path}/idx_remove.npy", idx_removed)

    print(f"Writing files for {obj_model}, dataset {dataset_number} ...")
    color_path = osp.join(data_path, "frames_color")
    os.makedirs(color_path)
    save_images(tactileImages=image, save_path=color_path)

    # clean data
    for idx in idx_removed:
        path = "{path}/frame_{p_i:07d}.jpg".format(path=color_path, p_i=idx)
        os.remove(path)
        path = "{path}/frame_{p_i:07d}.jpg".format(path=heightmap_path, p_i=idx)
        os.remove(path)
   
    ## uncomment for plotting
    # viz = Viz(off_screen=False)
    # viz.init_variables(mesh_path=obj_path, gt_pose=gt_p)
    # for i in range(len(hm)):
    #     viz.update(gt_pose=gt_p_cam[i+ini_idx], image=image[i], heightmap=hm[i], mask=cm[i], frame=i+ini_idx, image_savepath=None)
    # viz.close()


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    touch_simulator(cfg=cfg)


if __name__ == "__main__":
    main()