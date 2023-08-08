import bpy
from .materials import plane_mat  # noqa


def setup_cycles(cycle=True):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # bpy.context.scene.render.film_transparent = True

    if cycle:
        bpy.context.scene.cycles.use_denoising = True
    # added for python versions >3, I have 3.1.2(BLENDER)
    if bpy.app.version[0] == 3:
        if bpy.context.scene.cycles.device == "GPU":
            bpy.context.scene.cycles.tile_size = 256
        else:
            bpy.context.scene.cycles.tile_size = 32
    else:
        bpy.context.scene.render.tile_x = 256
        bpy.context.scene.render.tile_y = 256

    bpy.context.scene.cycles.samples = 64

    # bpy.context.scene.render.film_transparent = True

    # bpy.context.scene.cycles.denoiser = 'OPTIX'

def setup_eevee():

    for scene in bpy.data.scenes:
        scene.render.engine = 'BLENDER_EEVEE'
        # bpy.context.preferences.addons["eevee"].preferences.compute_device_type = "CUDA"
        # bpy.context.scene.eevee.device = "GPU"

        # Enable Eevee features
        scene = bpy.context.scene
        eevee = scene.eevee

        eevee.use_soft_shadows = True

        eevee.use_ssr = True
        eevee.use_ssr_refraction = True

        eevee.use_gtao = True
        eevee.gtao_distance = 1

        eevee.use_volumetric_shadows = True
        eevee.volumetric_tile_size = '2'

        for mat in bpy.data.materials:
            # This needs to be enabled case by case,
            # otherwise we loose SSR and GTAO everywhere.
            # mat.use_screen_refraction = True
            mat.use_sss_translucency = True

        cubemap = None
        grid = None
        # Does not work in edit mode
        invert = False
        try:
            # Simple probe setup
            bpy.ops.object.lightprobe_add(type='CUBEMAP', location=(0.5, 0, 1.5))
            cubemap = bpy.context.selected_objects[0]
            cubemap.scale = (2.5, 2.5, 1.0)
            cubemap.data.falloff = 0
            cubemap.data.clip_start = 2.4

            bpy.ops.object.lightprobe_add(type='GRID', location=(0, 0, 0.25))
            grid = bpy.context.selected_objects[0]
            grid.scale = (1.735, 1.735, 1.735)
            grid.data.grid_resolution_x = 3
            grid.data.grid_resolution_y = 3
            grid.data.grid_resolution_z = 2
        except:
            pass

        try:
            # Try to only include the plane in reflections
            plane = bpy.data.objects['Plane']

            collection = bpy.data.collections.new("Reflection")
            collection.objects.link(plane)
            # Add all lights to light the plane
            if not invert:
                for light in bpy.data.objects:
                    if light.type == 'LIGHT':
                        collection.objects.link(light)

            # Add collection to the scene
            scene.collection.children.link(collection)

            cubemap.data.visibility_collection = collection
        except:
            pass

        eevee.gi_diffuse_bounces = 1
        eevee.gi_cubemap_resolution = '128'
        eevee.gi_visibility_resolution = '16'
        eevee.gi_irradiance_smoothing = 0

        bpy.ops.scene.light_cache_bake()




# Setup scene
def setup_scene(cycle=True, res='low'):
    scene = bpy.data.scenes['Scene']
    assert res in ["ultra", "high", "med", "low"]
    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280//2
        scene.render.resolution_y = 1024//2
    elif res == "low":
        scene.render.resolution_x = 1280//4
        scene.render.resolution_y = 1024//4
    elif res == "ultra":
        scene.render.resolution_x = 1280*2
        scene.render.resolution_y = 1024*2
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.object.light_add(type='SUN', align='WORLD',
                             location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = 1.5

    # rotate camera
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.transform.resize(value=(10, 10, 10), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.object.select_all(action='DESELECT')

    # setup_cycles(cycle=cycle)
    setup_eevee()

    return scene
