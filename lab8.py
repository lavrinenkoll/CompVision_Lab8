import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d


# variables
# path to images
imgs = [
    # vertical img pair with different shift
    ["./imgs/-2147483648_-232454.jpg", "./imgs/-2147483648_-232456.jpg"], # low shift
    ["./imgs/-2147483648_-232458.jpg", "./imgs/-2147483648_-232460.jpg"],
    ["./imgs/-2147483648_-232462.jpg", "./imgs/-2147483648_-232464.jpg"],
    ["./imgs/-2147483648_-232468.jpg", "./imgs/-2147483648_-232470.jpg"], # high shift

    # horizontal img pair with different shift
    ["./imgs/photo_2023-12-25_18-39-43.jpg", "./imgs/photo_2023-12-25_18-39-43 (2).jpg"], # low shift
    ["./imgs/photo_2023-12-25_18-39-47.jpg", "./imgs/photo_2023-12-25_18-39-47 (2).jpg"], # high shift

    # vertical img pair, another object
    ["./imgs/-2147483648_-232482.jpg", "./imgs/-2147483648_-232484.jpg"],
]

# params for StereoSGBM_create for each pair of images
params = [
    {"minDisparity": 15, "numDisparities": 30, "blockSize": 16, "P1": 8 * 3 * 3 ** 2, "P2": 32 * 3 * 3 ** 2,
        "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},
    {"minDisparity": 30, "numDisparities": 100, "blockSize": 16, "P1": 8 * 3 * 3 ** 2, "P2": 32 * 3 * 3 ** 2,
        "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},
    {"minDisparity": 50, "numDisparities": 120, "blockSize": 16, "P1": 8 * 3 * 7 ** 2, "P2": 32 * 3 * 7 ** 2,
        "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},
    {"minDisparity": 80, "numDisparities": 190, "blockSize": 16, "P1": 8 * 3 * 7 ** 2, "P2": 32 * 3 * 7 ** 2,
     "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},

    {"minDisparity": 60, "numDisparities": 100, "blockSize": 16, "P1": 8 * 3 * 3 ** 2, "P2": 32 * 3 * 3 ** 2,
        "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},
    {"minDisparity": 95, "numDisparities": 150, "blockSize": 16, "P1": 8 * 3 * 3 ** 2, "P2": 32 * 3 * 3 ** 2,
     "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},

    {"minDisparity": 20, "numDisparities": 100, "blockSize": 16, "P1": 8 * 3 * 3 ** 2, "P2": 32 * 3 * 3 ** 2,
        "disp12MaxDiff": 1, "uniquenessRatio": 10, "speckleWindowSize": 100, "speckleRange": 32},
]

# ply header for 3d point cloud
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


# function for generating 3d point cloud from disparity map and saving it to .ply file
def generate_3d(img, disp, filename="out.ply"):
    # size of image
    h, w = img.shape[:2]
    # focal length
    f = 0.8 * w
    # projection matrix for 3d point cloud
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],
                    [0, 0, 0, -f],
                    [0, 0, 1, 0]])
    # generate 3d point cloud
    points = cv2.reprojectImageTo3D(disp, Q)
    # generate colors for 3d point cloud
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # mask for removing background
    mask = disp > disp.min()
    # apply mask
    out_points = points[mask]
    out_colors = colors[mask]

    # save 3d point cloud to .ply file
    out_points = out_points.reshape(-1, 3)
    out_colors = out_colors.reshape(-1, 3)
    out_points = np.hstack([out_points, out_colors])
    with open(filename, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(out_points))).encode('utf-8'))
        np.savetxt(f, out_points, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    # create folders for saving 3d point clouds
    try:
        os.mkdir("result_models")
    except OSError:
        # recreate folders if exist
        shutil.rmtree("result_models")
        os.mkdir("result_models")

    # main loop for processing images
    for i in range(len(imgs)):
        # load images
        img1 = cv2.pyrDown(cv2.imread(imgs[i][0]))
        img2 = cv2.pyrDown(cv2.imread(imgs[i][1]))

        # save original images
        img1_save = img1.copy()
        img2_save = img2.copy()

        # blur images
        img1 = cv2.blur(img1, (5, 5))
        img2 = cv2.blur(img2, (5, 5))

        # convert images to grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # create stereo matcher
        stereo = cv2.StereoSGBM_create(**params[i])

        # compute disparity map
        disp = stereo.compute(img1, img2).astype(np.float32) / 16.0

        # show images and disparity map
        img_all = np.hstack((img1_save, img2_save))
        #cv2.imshow("imgs", img_all)
        plt.imshow(img_all)
        plt.title("imgs"+str(i+1))
        plt.show()
        #cv2.imshow("disparity", (disp - params[i]["minDisparity"]) / params[i]["numDisparities"])
        plt.imshow((disp - params[i]["minDisparity"]) / params[i]["numDisparities"])
        plt.title("disparity"+str(i+1))
        plt.show()

        # generate 3d point cloud
        generate_3d(img1_save, disp, f"./result_models/out{i+1}.ply")

        # show 3d point cloud
        view = o3d.io.read_point_cloud(f"./result_models/out{i+1}.ply")
        o3d.visualization.draw_geometries([view], window_name=f"out{i+1}.ply", width=800, height=800)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
