import os
import random
import numpy as np
import geopandas as gpd # Handles geospatial vector data
import osmnx as ox # OSM street networks
from osmnx.projection import project_gdf
import rasterio # Rasterize vector geometries
from rasterio.features import rasterize
from shapely.geometry import LineString, Point, mapping # manipulate geometric objects
from PIL import Image, ImageDraw # Saves raster images
from scipy.ndimage import gaussian_filter
import json
from tqdm import tqdm

# Reference:
# Augmentation : https://www.datacamp.com/tutorial/complete-guide-data-augmentation

# Keep parameters bundled
CONFIG = 'oxford-town'

CONFIGS = {
    "oxford-town": {
        "place": "Oxford, Ohio, USA",
        "image_size": (256, 256),
        "n_samples": 450,
        "network_type": "drive",  
        'data_dir': 'data/thinning/Oxford',
        "buffer_m": 128,  # Sets radius around interection point
        "augmentation": {
            # Gaussian Noise std (alters pixel values randomly to simulate environment inperfections)
            "noise_std": 10,
            # Guassian Blur std (Smooths image, reduces detail), simulates low res images
            "blur_sigma": 2,
            # Adds random rotation to images (-d to +d degrees), simulates different map orientation
            # Decided to not use rotation
            # "rotation_range": 30
        }        
    },
}

for c in CONFIGS:
  CONFIGS[c]['name'] = c 

# City of Oxford has these road types
# ['residential' 'tertiary' 'unclassified' 'trunk' 'primary' 'secondary' 'trunk_link']

# Thickness by road type (in meters)
# formatted as (lower, upper)
DEFAULT_THICKNESS = {
    'trunk': (15, 30),
    'primary': (10, 25),
    'secondary': (7, 15),
    'tertiary': (5, 10),
    'unclassified': (3, 7),
    'residential': (4, 8),
    'trunk_link': (6, 12),
}

def get_default_thickness(row):
    tag = row.get('highway', 'residential')
    if isinstance(tag, list):
        tag = tag[0]  # pick the first tag if it's a list
    lower, upper =  DEFAULT_THICKNESS.get(tag, (2, 4))
    return random.uniform(lower, upper)


def download_osm_data(config, cache_dir="cache"):
    config_name = config.get("name", "default")
    place = config["place"]
    network_type = config.get("network_type", "all")

    config_cache_dir = os.path.join(cache_dir, config_name)
    os.makedirs(config_cache_dir, exist_ok=True)

    edges_fp = os.path.join(config_cache_dir, "edges.gpkg")
    nodes_fp = os.path.join(config_cache_dir, "nodes.gpkg")

    if os.path.exists(edges_fp) and os.path.exists(nodes_fp):
        print(f"Loading cached OSM data for {place}...")
        gdf_edges = gpd.read_file(edges_fp)
        gdf_nodes = gpd.read_file(nodes_fp)
    else:
        print(f"Fetching roads from {place}...")
        G = ox.graph_from_place(place, network_type=network_type)
        print("Converting to dataframes...")
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
        
        print("Projecting to UTM ...")
        gdf_edges = project_gdf(gdf_edges) # Automatically choosed a UTM zone
        gdf_nodes = project_gdf(gdf_nodes)

        print(f"Fetched {len(gdf_edges)} edges and {len(gdf_nodes)} nodes.")
        gdf_edges.to_file(edges_fp, driver="GPKG")
        gdf_nodes.to_file(nodes_fp, driver="GPKG")

    return gdf_edges, gdf_nodes

def pick_random_intersections(nodes, n=500):
    return nodes.sample(n)

def rasterize_roads(roads, bounds, size, thickness_fn):
    """
    Rasterize roads into a binary mask using rasterio.
    Thickness is applied by buffering each LineString in meters.
    """
    # Create transform (affine mapping from pixel coords to geographic coords)
    transform = rasterio.transform.from_bounds(*bounds, width=size[0], height=size[1])

    # Buffer each line to simulate thickness and generate (geometry, value) tuples
    shapes = []
    for _, row in roads.iterrows():
        geom = row.geometry
        if geom is None or not isinstance(geom, LineString):
            continue
        thickness = thickness_fn(row) / 2.0  # buffer radius in meters
        if thickness > 0:
            buffered = geom.buffer(thickness)
            shapes.append((buffered, 255))  # Burn value = 1 for road
        else:
            shapes.append((geom, 255))

    mask = rasterize(
        shapes=shapes,
        out_shape=size,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask

def generate_samples(gdf_edges, gdf_nodes, n_samples=450, out_dir="dataset", image_size=(256, 256)):
    os.makedirs(out_dir, exist_ok=True)

    # Create two subfolders for images and skeletons
    images_dir = os.path.join(out_dir, "images")
    skeletons_dir = os.path.join(out_dir, "skeletons")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(skeletons_dir, exist_ok=True)

    selected_nodes = pick_random_intersections(gdf_nodes, n=n_samples)

    for i, (_, node) in enumerate(tqdm(selected_nodes.iterrows(), total=n_samples)):
        pt = node.geometry
        buffer_m = 128  # meters around the point
        bounds = (pt.x - buffer_m, pt.y - buffer_m, pt.x + buffer_m, pt.y + buffer_m)
        clip = gdf_edges.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
        if clip.empty:
            continue

        image_array = rasterize_roads(clip, bounds, image_size, get_default_thickness)
        target_array = rasterize_roads(clip, bounds, image_size, lambda row: 0.0)

        # --- START: augmentation ---
        image_array = gaussian_filter(image_array, sigma=CONFIGS[CONFIG]['augmentation']['blur_sigma'])
        noise = np.random.normal(0, CONFIGS[CONFIG]['augmentation']['noise_std'], image_array.shape)
        image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        # --- END: augmentation ---

        # Save image in images/ folder
        image_path = os.path.join(images_dir, f"image_{i:05d}.png")
        Image.fromarray(image_array).save(image_path)

        # Save skeleton in skeletons/ folder
        skeleton_path = os.path.join(skeletons_dir, f"skeleton_{i:05d}.png")
        Image.fromarray(target_array).save(skeleton_path)

        # Save geojson (unchanged)
        target_geojson_path = os.path.join(out_dir, f"target_{i:05d}.geojson")
        features = [{
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {"highway": row.get("highway", "unknown")}
        } for _, row in clip.iterrows() if isinstance((geom := row.geometry), LineString)]
        
        geojson = {"type": "FeatureCollection", "features": features}
        with open(target_geojson_path, 'w') as f:
            json.dump(geojson, f)

# Filter by subset instead of all drivable paths
def filter_edges_by_highway(gdf_edges, accepted_types):
    def highway_filter(tags):
        if isinstance(tags, list):
            return any(tag in accepted_types for tag in tags)
        else:
            return tags in accepted_types

    return gdf_edges[gdf_edges['highway'].apply(highway_filter)]

if __name__ == "__main__":
  config=CONFIGS[CONFIG]
  edges, nodes = download_osm_data(config)
  # print(edges['highway'].unique())

  # Filter to subset by removing unclassifed road types
  subset_types = ['residential', 'tertiary', 'trunk', 'primary', 'secondary' 'trunk_link']
  edges_filtered = filter_edges_by_highway(edges, subset_types)
  
  generate_samples(edges_filtered, nodes, n_samples=config['n_samples'], out_dir=config.setdefault('data_dir', './data/thinning'))
