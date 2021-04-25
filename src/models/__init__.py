from core.dataset import lidar_dataset, coord_dataset, coord_lidar_dataset, coord_lidar_image_dataset, image_dataset
from models.beamsoup import beamsoup_coord_model_fn, beamsoup_joint_model_fn, beamsoup_lidar_model_fn
from models.imperial import imperial_model_fn
from models.nu_huskies import nu_husky_coord_model_fn, nu_husky_lidar_model_fn, nu_husky_image_model_fn, \
    nu_husky_fusion_model_fn
from models.southampton import southampton_model_fn

models = {
    'imperial': (imperial_model_fn, lidar_dataset),

    'beamsoup-coord': (beamsoup_coord_model_fn, coord_dataset),
    'beamsoup-lidar': (beamsoup_lidar_model_fn, lidar_dataset),
    'beamsoup-joint': (beamsoup_joint_model_fn, coord_lidar_dataset),

    'husky-coord': (nu_husky_coord_model_fn, coord_dataset),
    'husky-lidar': (nu_husky_lidar_model_fn, lidar_dataset),
    'husky-image': (nu_husky_image_model_fn, image_dataset),
    'husky-fusion': (nu_husky_fusion_model_fn, coord_lidar_image_dataset),

    'southampton': (southampton_model_fn, coord_lidar_dataset)
}
