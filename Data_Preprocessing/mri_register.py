import ants

def register_image(moving_image, fixed_image):

    mytx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine' ,grad_step=0.1)

    mywarpedimage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=mytx['fwdtransforms'])

    return mywarpedimage