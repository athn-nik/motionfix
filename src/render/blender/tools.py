import bpy
import numpy as np


def mesh_detect(data):
    # heuristic
    if data.shape[1] > 1000:
        return True
    return False


# see this for more explanation
# https://gist.github.com/iyadahmed/7c7c0fae03c40bd87e75dc7059e35377
# This should be solved with new version of blender
class ndarray_pydata(np.ndarray):
    def __bool__(self) -> bool:
        return len(self) > 0


# body parts
bp_smplh_parts = {
    'global': ['hips'],
    'torso': ['spine1', 'spine2', 'head', 'neck', 'spine'],
    'left arm': ['leftArm', 'leftShoulder', 'leftHandIndex1', 'leftForeArm', 'leftHand'],
    'right arm': ['rightHand', 'rightShoulder', 'rightArm', 'rightHandIndex1', 'rightForeArm'],
    'left leg': ['leftLeg', 'leftToeBase', 'leftFoot', 'leftUpLeg'],
    'right leg': ['rightUpLeg', 'rightFoot', 'rightLeg', 'rightToeBase']
}

bp_priority = {
    'global': True,
    'torso': False,
    'left arm': True,
    'right arm': True,
    'left leg': False,
    'right leg': False
}



smplh_parts_to_bp = {}
for key, vals in bp_smplh_parts.items():
    for val in vals:
        smplh_parts_to_bp[val] = key


def load_numpy_vertices_into_blender(vertices, faces, name, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces.view(ndarray_pydata))
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    obj.active_material = mat
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action='DESELECT')
    return True


def load_numpy_vertices_into_blender_bp(vertices, faces, name, bp_list, mats, default_mat, jpath='deps/inference/smpl_part_seg.json'):
    import json
    segmentation_map = json.load(open(jpath))
    bps = list(bp_smplh_parts.keys())
    bp_vtx = {bp: [] for bp in bps}

    for smplh_bp, vtx in segmentation_map.items():
        bp_vtx[smplh_parts_to_bp[smplh_bp]].append(vtx)

    # flatten
    for bp in bps:
        bp_vtx[bp] = [y for x in bp_vtx[bp] for y in x]

    # fix overlaps
    bp_vtx["left arm"] = list(set(bp_vtx["left arm"]).difference(set(bp_vtx["torso"])))
    bp_vtx["right arm"] = list(set(bp_vtx["right arm"]).difference(set(bp_vtx["torso"])))
    bp_vtx["torso"] = list(set(bp_vtx["torso"]).difference(set(bp_vtx["global"])))
    bp_vtx["left leg"] = list(set(bp_vtx["left leg"]).difference(set(bp_vtx["global"])))
    bp_vtx["right leg"] = list(set(bp_vtx["right leg"]).difference(set(bp_vtx["global"])))

    for bp, mat, bp_choosen in zip(bps, mats, bp_list):
        if not bp_choosen:
            mat = default_mat

        bp_name = bp + "_" + name
        mesh = bpy.data.meshes.new(bp_name)

        mask = np.zeros(len(vertices), dtype=bool)
        mask[bp_vtx[bp]] = True

        verts_bp = vertices[mask]
        choosen = np.where(mask)[0]
        discarded = np.where(~mask)[0]

        from einops import rearrange
        # filter = np.in1d(rearrange(faces, "i j -> (i j)"), discarded)
        # filter = rearrange(filter, "(i j) -> i j", j=3)
        # filter = filter.any(1)
        # import ipdb; ipdb.set_trace()
        # filter = filter.all(1)

        if True:
            filter = np.in1d(rearrange(faces, "i j -> (i j)"), choosen)
            filter = rearrange(filter, "(i j) -> i j", j=3)
            if bp_priority[bp]:
                filter = filter.any(1)
            else:
                filter = filter.all(1)

            non_mapped_faces_bp = faces[filter]
            vtx_id_to_add = set(np.unique(non_mapped_faces_bp)).difference(set(choosen))
            verts_bp = np.concatenate((verts_bp, vertices[list(vtx_id_to_add)]))
            all_vtx = np.concatenate((choosen, np.array(list(vtx_id_to_add))))
        else:
            # remove other vertices
            filter = np.in1d(rearrange(faces, "i j -> (i j)"), discarded)
            filter = rearrange(filter, "(i j) -> i j", j=3)
            filter = filter.any(1)

            choosen = np.unique(list(set(choosen).difference(np.unique(faces[filter]))))
            verts_bp = vertices[choosen]

            # take only the good ones
            filter = np.in1d(rearrange(faces, "i j -> (i j)"), choosen)
            filter = rearrange(filter, "(i j) -> i j", j=3)
            filter = filter.all(1)

            non_mapped_faces_bp = faces[filter]
            all_vtx = choosen

        faces_bp = np.zeros_like(non_mapped_faces_bp)
        for idx, x in enumerate(all_vtx):
            sub_mask = non_mapped_faces_bp == x
            faces_bp[sub_mask] = idx

        mesh.from_pydata(verts_bp, [], faces_bp.view(ndarray_pydata))
        mesh.validate()

        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        obj.active_material = mat
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        bpy.ops.object.select_all(action='DESELECT')

    return True


def delete_objs(names):
    if not isinstance(names, list):
        names = [names]
    # bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        for name in names:
            if obj.name.startswith(name) or obj.name.endswith(name):
                obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
