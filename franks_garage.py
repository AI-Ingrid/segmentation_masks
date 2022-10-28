from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks


def load_franks_vehicles():
    franks_car_1 = read_image("/Users/ikolderu/PycharmProjects/visualize_seg_masks/franks_car_1.jpeg")
    franks_car_2 = read_image("/Users/ikolderu/PycharmProjects/visualize_seg_masks/franks_car_2.jpeg")
    franks_car_3 = read_image("/Users/ikolderu/PycharmProjects/visualize_seg_masks/franks_car_3.jpeg")

    franks_motorbike_1 = read_image("/Users/ikolderu/PycharmProjects/visualize_seg_masks/franks_motorbike_1.jpeg")
    franks_motorbike_2 = read_image("/Users/ikolderu/PycharmProjects/visualize_seg_masks/franks_motorbike_2.jpeg")
    franks_motorbike_3 = read_image("/Users/ikolderu/PycharmProjects/visualize_seg_masks/franks_motorbike_3.jpeg")

    return [franks_car_1, franks_car_2, franks_car_3, franks_motorbike_1, franks_motorbike_2, franks_motorbike_3]


def reshape_the_images(images):
    transform = torchvision.transforms.Resize((256, 256))
    reshaped_images = []

    for image in images:
        reshaped_image = transform(image)
        reshaped_images.append(reshaped_image)

    return reshaped_images


def show_images(images):
    if not isinstance(images, list):
        images = [images]

    fig, axs = plt.subplots(ncols=len(images), squeeze=False)

    for index, image in enumerate(images):
        image = image.detach()
        image = F.to_pil_image(image)
        axs[0, index].imshow(np.asarray(image))
        axs[0, index].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def get_trained_network():
    weights = FCN_ResNet50_Weights.DEFAULT
    transforms = weights.transforms(resize_size=None)

    model = fcn_resnet50(weights=weights, progress=False)
    model = model.eval()
    return transforms, model, weights


def forward(transforms, images, model):
    batch = torch.stack([transforms(image) for image in images])
    output = model(batch)['out']
    return output


def add_segmentation_masks(weights, network_output, images):
    sem_class_to_idx = {cls: index for (index, cls) in enumerate(weights.meta["categories"])}

    normalized_masks = torch.nn.functional.softmax(network_output, dim=1)

    masks = [
        normalized_masks[image_index, sem_class_to_idx[cls]]
        for image_index in range(len(images))
        for cls in ('car', 'motorbike')
    ]
    return masks, normalized_masks, sem_class_to_idx


def get_boolean_mask(normalized_masks, segmentation_class):
    class_dim = 1
    boolean_masks = (normalized_masks.argmax(class_dim) == segmentation_class)
    print(f"shape = {boolean_masks.shape}, dtype = {boolean_masks.dtype}")
    return boolean_masks


def plot_segmentation_masks_on_images(images, boolean_masks):
    images_with_masks = [draw_segmentation_masks(image, masks=mask, alpha=0.7, colors="magenta")
                         for image, mask in zip(images, boolean_masks)]
    show_images(images_with_masks)
    return images_with_masks


def plot_all_segmentation_masks_on_image(normalized_masks, image):
    num_classes = normalized_masks.shape[1]
    image_masks = normalized_masks[0]
    class_dim = 0
    colors = ["cyan", "pink", "blue", "green", "white", "black", "plum",
              "magenta", "red", "gold", "purple", "orange", "lightseagreen", "darkred",
              "navy", "lime", "indigo", "green", "coral", "darkcyan", "blueviolet"]
    image_all_classes_masks = image_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
    image_with_all_masks = draw_segmentation_masks(image, masks=image_all_classes_masks, alpha=0.6, colors=colors)
    show_images(image_with_all_masks)


def plot_all_segmentation_masks_on_images(normalized_masks, images):
    num_classes = normalized_masks.shape[1]
    class_dim = 1
    all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
    all_classes_masks = all_classes_masks.swapaxes(0, 1)
    colors = ["cyan", "pink", "blue", "green", "white", "black", "plum",
              "magenta", "red", "gold", "purple", "orange", "lightseagreen", "darkred",
              "navy", "lime", "indigo", "green", "coral", "darkcyan", "blueviolet"]
    images_with_masks = [draw_segmentation_masks(image, masks=mask, alpha=0.6, colors=colors)
                         for image, mask in zip(images, all_classes_masks)]
    show_images(images_with_masks)


def main():
    # Load Frank's Garage
    franks_vehicles = load_franks_vehicles()
    franks_vehicles = reshape_the_images(franks_vehicles)

    # Show Frank's vehicles
    show_images(franks_vehicles)

    # Get trained network and send images in the network
    transforms, model, weights = get_trained_network()
    network_output = forward(transforms, franks_vehicles, model)

    # Add segmentation masks on the images
    franks_vehicles_masks, normalized_masks, sem_class_to_index = add_segmentation_masks(weights, network_output, franks_vehicles)

    # Show images with segmentations masks
    show_images(franks_vehicles_masks)

    # Get boolean segmentation masks
    boolean_car_masks = get_boolean_mask(normalized_masks, sem_class_to_index['car'])
    show_images([mask.float() for mask in boolean_car_masks[:3]])
    boolean_motorbike_masks = get_boolean_mask(normalized_masks, sem_class_to_index['motorbike'])
    show_images([mask.float() for mask in boolean_motorbike_masks[3:]])

    # Plot masks on top of images
    plot_segmentation_masks_on_images(franks_vehicles[3:], boolean_motorbike_masks[3:])
    plot_segmentation_masks_on_images(franks_vehicles[:3], boolean_car_masks[:3])

    print("type", type(sem_class_to_index))
    print(sem_class_to_index)
    classes = list(sem_class_to_index.keys())
    colors = ["cyan", "pink", "blue", "green", "white", "black", "plum",
              "magenta", "red", "gold", "purple", "orange", "lightseagreen", "darkred",
              "navy", "lime", "indigo", "green", "coral", "darkcyan", "blueviolet"]

    for i in range(len(classes)):
        print("Class: ", classes[i], " Color: ", colors[i])

    # Plot all masks on top of all images
    plot_all_segmentation_masks_on_images(normalized_masks, franks_vehicles)


main()
