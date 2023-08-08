import sys
if 'blender' not in sys.executable:
    import sys
    from .anim import render_animation
    from .mesh_viz import visualize_meshes
