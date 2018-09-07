from scipy import misc
from utils.anisotropic_diffusion import perona_malik


def preprocess(img_dir, datas):
    for data in datas:
        img_path = "{}/{}".format(img_dir, data[0])

        # Image Conversion
        img = misc.imread(img_path, flatten=True)
        img = img.astype('float32')

        iterations = 30
        delta = 0.14
        kappa = 15

        img, gradient = perona_malik(img, iterations, delta, kappa)

        file_name = data[0].split('.')[0]
        preprocessed_path = "knots_processed/{}.png".format(file_name)
        misc.imsave(preprocessed_path, img)
