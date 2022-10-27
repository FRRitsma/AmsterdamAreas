# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import transforms


def unpack(input_list: list) -> list:
    """
    Transforms all arrays in a nested list from list format to numpy
    """
    if type(input_list) is not list:
        return input_list
    output_list = []
    for i in input_list:
        if len(np.array(i).shape) == 2:
            output_list.append(np.array(i))
        else:
            output_list.append(unpack(i))
    return output_list


def flatten(input_list: list) -> list:
    """
    Transforms a nested list into a flattened list
    """
    if type(input_list) is not list:
        return input_list
    flat_list = []
    for item in input_list:
        if type(item) is not list:
            flat_list.append(item)
        else:
            for sub_item in flatten(item):
                flat_list.append(sub_item)
    return flat_list


def rotate(array: np.ndarray, theta: float, point: np.ndarray) -> np.ndarray:
    """
    Rotates array by angle theta around point
    """
    array = array - point
    theta = (theta / 180) * np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    array = array @ R
    array = array + point
    return array


def plot_area_of_amsterdam(
    df: pd.core.frame.DataFrame,
    title: str = "Neighborhoods of Amsterdams",
    color_mapping=None,
) -> None:
    """
    Plot neighborhouds of Amsterdam
    """
    # Get min-max range of coordinates:
    minx = np.array([1, 1]) * float("inf")
    maxx = -minx
    for idx, row in df.iterrows():
        for poly in row["polygon"]:
            minx[0] = min(minx[0], min(poly[:, 0]))
            minx[1] = min(minx[1], min(poly[:, 1]))
            maxx[0] = max(maxx[0], max(poly[:, 0]))
            maxx[1] = max(maxx[1], max(poly[:, 1]))

    difx = maxx - minx
    maxx += difx * 0.1
    minx -= difx * 0.1
    difx = maxx - minx

    # Add polygons to plot:
    fig, ax = plt.subplots()
    patches = []
    colors = []
    for _, row in df.iterrows():
        for poly in row["polygon"]:
            poly = (poly - minx) / (maxx - minx)
            poly[:, 0] = abs(1 - poly[:, 0])
            poly = rotate(poly, 90, 0.5 * difx)
            if color_mapping is not None:
                colors.append(color_mapping(poly))
            else:
                colors.append(float(np.random.rand(1)))
            polygon = Polygon(poly, closed=True)
            patches.append(polygon)
    p = PatchCollection(patches, alpha=0.9)
    colors = np.array(colors)
    # Translate in image:
    minus_shift = transforms.Affine2D().translate(difx[0], 1.5 * difx[1])\
        + ax.transData
    p.set_transform(minus_shift)
    # Color the patches:
    p.set_array(colors)
    # Add patches:
    ax.add_collection(p)
    # Set title:
    ax.set_title(title)
    plt.show()


# %%


with open("poly.json", "r") as fs:
    geodata = json.load(fs)

features = geodata["features"]
df = pd.DataFrame(features)

df["polygon"] = df["geometry"].apply(lambda x: x["coordinates"])
df["polygon"] = df["polygon"].apply(unpack)
df["polygon"] = df["polygon"].apply(flatten)

plot_area_of_amsterdam(df)

# %%
