from matplotlib import pyplot as plt
from run_constants import *
from yumi_jacobi.interface import Interface
from motion_jacobi import goto_gripper_px, _raise_to_safe_height_and_home

## if this doesnt' work, switch BrioSensor port to 1/0


def click_points_simple(img):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    left_coords, right_coords = None, None

    def onclick(event):
        xind, yind = int(event.xdata), int(event.ydata)
        coords = (xind, yind)
        nonlocal left_coords, right_coords
        if event.button == 1:
            left_coords = coords
        elif event.button == 3:
            right_coords = coords

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Left / Right click on point, press Q when done")
    plt.tight_layout()
    plt.show()
    return left_coords, right_coords


cam = BRIOSensor(0)
interface = Interface(speed=0.5)
interface.yumi.left.min_position  = YUMI_MIN_POS
interface.yumi.right.min_position = YUMI_MIN_POS

process = True
arm = "placeholder"

if __name__ == "__main__":
    while True:
        input("Press Enter: ")
        if process:
            _raise_to_safe_height_and_home(interface, "left")
            _raise_to_safe_height_and_home(interface, "right")
            process = False
        else:
            _raise_to_safe_height_and_home(interface, arm)
        
        for _ in range(5):
            img = cam.read()

        point, _ = click_points_simple(img)
        print(f"shape {img.shape}")
        plt.figure(figsize=(20, 12))
        plt.scatter(point[0], point[1])
        plt.tight_layout()
        plt.imshow(img)
        plt.show()
        arm = goto_gripper_px(point, interface, hover_only=True)
