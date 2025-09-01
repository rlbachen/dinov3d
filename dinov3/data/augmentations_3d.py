import logging
import torch
from dinov3.data.transforms_3d import RandomResizedCrop3d

from monai.transforms import (
    Compose,
    Resize,
    LoadImaged,
    RandFlip,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandBiasField,
    RandHistogramShift,
    RandGibbsNoise,
    ToTensor,
    Spacingd,
    DivisiblePadd,
    MapTransform
)


logger = logging.getLogger("dinov3")

def _to_3tuple(x):
    if isinstance(x, (list, tuple)):
        if len(x) != 3:
            raise ValueError(f"Expected 3D size (len 3), got {x}")
        return tuple(x)
    return (x, x, x)

class DataAugmentation3d(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        keys: list[str] = ["image"],
        global_crops_size=96,
        local_crops_size=48,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        flips=True,  # renaming to flips, as we flip aroud all spatial axis equally
    ):
        """
        This class is a wrapper around the monai transforms to apply data augmentation to a 3D MRI image.

        make sure the global and local transforms are non-dictonary transforms.
        """

        if len(keys) != 1:
            raise ValueError(f"DataAugmentation3d only supports a singlekey, recived: {keys}")
        else:
            self.key = keys[0]

        if local_crops_subset_of_global_crops:
            # this argument is not used in any of the training stages configs here, so we will skip it
            raise NotImplementedError("Local crops subset of global crops not implemented for 3d")

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = _to_3tuple(global_crops_size)
        self.local_crops_size = _to_3tuple(local_crops_size)
        self.gram_teacher_crops_size = (_to_3tuple(gram_teacher_crops_size)
                                        if gram_teacher_crops_size is not None else None)
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size

        if local_crops_subset_of_global_crops:
            raise NotImplementedError(
                "local crop subset of global crops not implement for 3d, as it is not used in training"
            )

        logger.info("###################################")
        logger.info("Using 3d data augmentation parameters:")
        logger.info(f"keys: {keys}")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"flips : {flips}")
        logger.info("###################################")

        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # here we separate cropping and other transforms to keep track of spacing
        self.global_cropper = RandomResizedCrop3d(crop_size=global_crop_max_size, scale=global_crops_scale)

        self.geometric_augmentation_global = Compose([RandFlip(prob=0.5 if flips else 0.0)])

        resize_global = torch.nn.Identity()
        self.resize_global_post_transf = torch.nn.Identity()
        self.resize_gram_teacher = None

        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = Resize(self.global_crops_size)
            else:
                self.resize_global_post_transf = Resize(self.global_crops_size)

        self.resize_gram_teacher = Resize(
            self.gram_teacher_crops_size  # maybe set interpolation method
        )

        self.local_cropper = RandomResizedCrop3d(crop_size=self.local_crops_size, scale=self.local_crops_scale)

        self.geometric_augmentation_local = Compose(
            [
                RandGaussianSmooth(prob=0.1),
                RandGaussianNoise(prob=0.1),
                RandBiasField(prob=0.1),
                RandHistogramShift(prob=0.1),
                ToTensor(),
                RandFlip(prob=0.5 if flips else 0.0),
            ]
        )

        global_transfo1_extra = Compose(
            [RandGaussianSmooth(prob=1), RandGaussianNoise(prob=0.1), RandGibbsNoise(prob=0.2)]
        )

        global_transfo2_extra = Compose(
            [RandGaussianSmooth(prob=0.1), RandGaussianNoise(prob=0.2), RandGibbsNoise(prob=0.1)]
        )

        local_transfo_extra = RandGaussianSmooth(prob=0.5)

        self.normalize = ToTensor()
        self.inital_transform = Compose(
            [
                LoadImaged(keys=[self.key], ensure_channel_first=True),
            ]
        )

        self.global_transfo1 = Compose([resize_global, global_transfo1_extra, self.normalize])
        self.global_transfo2 = Compose([resize_global, global_transfo2_extra, self.normalize])
        self.local_transfo = Compose([local_transfo_extra, self.normalize])

    def __call__(self, image_dict):
        output = {}
        output["weak_flag"] = True  # not sure what this does

        image_dict = self.inital_transform(image_dict)
        spacing = image_dict["spacing"]
        image = image_dict[self.key]

        im1_base = self.global_cropper(image, spacing)
        im1_base = self.geometric_augmentation_global(im1_base)  # maybe needs spacing arg
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.global_cropper(image, spacing)
        im2_base = self.geometric_augmentation_global(im2_base)  # maybe needs spacing arg
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                # Gram teacher gets resized directly from base crop before extra distortions.
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)

            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(self.local_cropper(image, spacing)))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


#making a separate inferance transform class, not sure if this is the right approach here
class InferenceAugmentation3d(object):
    def __init__(
        self,
        spacing: 1, # spacing in mm for isotropic resamping.
        patch_size = 16,
        keys: list[str] = ["image"]
    ):
        self.patch_size = patch_size
        self.keys = keys

        self.transform = Compose([
            LoadImaged(keys=self.keys, ensure_channel_first=True),
            Spacingd(
                keys=self.keys,
                pixdim=1,
                mode="bilinear", #not sure if this is equivalent to the torch trilinear used during inference, hmm
                padding_mode="border" # maybe zeros better not sure.
            ),
            DivisiblePadd(
                keys=self.keys,
                k = self.patch_size,
                mode = "minimum" # maybe use edge
            ),
        ])

    def __call__(self,data):
        return self.transform(data)