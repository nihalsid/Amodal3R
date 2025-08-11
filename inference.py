import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.

import sys
from pathlib import Path

# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import trimesh
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import postprocessing_utils
from PIL import Image


def save_mesh(mesh_result, filename):
    vertices = (
        mesh_result.vertices.cpu().numpy()
        if hasattr(mesh_result.vertices, "cpu")
        else mesh_result.vertices
    )
    faces = (
        mesh_result.faces.cpu().numpy()
        if hasattr(mesh_result.faces, "cpu")
        else mesh_result.faces
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if mesh_result.vertex_attrs is not None:
        attrs = (
            mesh_result.vertex_attrs.cpu().numpy()
            if hasattr(mesh_result.vertex_attrs, "cpu")
            else mesh_result.vertex_attrs
        )
        mesh.visual.vertex_colors = attrs

    mesh.export(filename)


# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
pipeline.cuda()


output_dir = str(Path(sys.argv[3]).parent)
os.makedirs(output_dir, exist_ok=True)

# can be single image or multiple images
images = [
    Image.open(sys.argv[1]).convert("RGB"),
]

masks = [
    Image.open(sys.argv[2]).convert("L"),
]


# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    masks,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)

mesh = outputs["mesh"][0]

# # save mesh
save_mesh(mesh, sys.argv[3])

# export glb if needed
# glb_path = os.path.join(output_dir, "mesh.glb")
# extract_glb(outputs['gaussian'][0], outputs['mesh'][0], 0.5, 1024, glb_path)
