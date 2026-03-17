import numpy as np
import matplotlib.pyplot as plt
import scipy


def click_points_simple(img):
    fig, (ax1) = plt.subplots(1,1)
    ax1.imshow(img)
    left_coords,right_coords = [],[]
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords.append(coords)
        elif(event.button==3):
            right_coords.append(coords)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords



def get_matching(old_endpts, new_endpts, viz=False):
    """ Generates mapping from the old set of endpoints to the new set of endpoints """
    # make matrix
    distances = scipy.spatial.distance_matrix(old_endpts, new_endpts)
    # run algorithm - N.B. This should be a mapping from old -> new
    old_indices, new_indices = scipy.optimize.linear_sum_assignment(distances)

    if viz:

        # display matchings
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

        # Display the first image with points
        ax1.imshow(old_image)
        for i, (x, y) in enumerate(old_endpts):
            ax1.scatter(x, y, color='red', marker='o')
            ax1.text(x + 3, y + 3, old_indices[i], color='red', fontsize=12)
        ax1.set_title('Old endpoints')

        # Display the second image with points
        ax2.imshow(new_image)
        for i, (x, y) in enumerate(new_endpts):
            ax2.scatter(x, y, color='red', marker='o')
            ax2.text(x + 3, y + 3, new_indices[i], color='red', fontsize=12)
        ax2.set_title('New endpoints')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the figure
        plt.show()

    return old_indices, new_indices # should be able to just use this


if __name__ == "__main__":
    # get an image
    old_image = np.load('../data/rope_knot_images_npy/knot_challenging_distractors/step_1/img_1693426335.3696506.npy')
    new_image = np.load('../data/rope_knot_images_npy/knot_challenging_distractors/step_1/img_1693426335.3696506.npy')

    # prompt click points (click all endpoints)
    old_endpts, _ = click_points_simple(old_image)
    new_endpts, _ = click_points_simple(new_image)

    for i, elem in enumerate(old_endpts):
        old_endpts[i] = np.array(elem)
    for i, elem in enumerate(new_endpts):
        new_endpts[i] = np.array(elem)

    old_indices, new_indices = get_matching(old_endpts, new_endpts, viz=True)

    