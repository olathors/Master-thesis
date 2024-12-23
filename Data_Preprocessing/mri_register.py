import ants

def register_image(moving_image, fixed_image):

    '''
    Function for registering mri image to atlas image.

    --PARAMETERS--

    moving_image: Image to be registered

    fixed_image: Atlas image
    
    returns: Registered ants image
    
    '''

    mytx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine' ,grad_step=0.1)

    mywarpedimage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=mytx['fwdtransforms'])

    return mywarpedimage