import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

epi_img = nib.load('someones_epi.nii.gz')
epi_img_data = epi_img.get_fdata()
print(epi_img_data.shape)


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")

img=mpimg.imread('10000104_1_CTce_ThAb_356.png')
imgplot = plt.imshow(img)
plt.show()
