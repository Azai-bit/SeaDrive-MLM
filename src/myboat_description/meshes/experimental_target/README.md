Place the experimental mesh for target_boat_2 in this directory.

Default file expected by the xacro:

- target2.obj

Default package URI used by the xacro:

- package://myboat_description/meshes/experimental_target2/target2.obj

The OBJ should reference the material file with a relative path, for example:

- mtllib material.mtl

The MTL should reference texture files in the same directory with relative paths, for example:

- map_Kd texture_pbr_20250901.png

If you use a different filename or extension, update:

- src/myboat_description/urdf_experimental/target2_experimental_mesh.xacro

Recommended starting assumptions:

- Units in meters
- Bow facing +X
- Up axis +Z
- Model origin near the hull centerline and close to the waterline