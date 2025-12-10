# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

import bpy
import bpy_extras.anim_utils
import mathutils
import math
import itertools
import re


# ------------------------------------------------------------------------
#  Action / channelbag helpers (used only for property FCurves)
# ------------------------------------------------------------------------

def _unwrap_action(action_ref):
    if action_ref is None:
        return None
    return action_ref.action if hasattr(action_ref, "action") else action_ref


def _get_action_slot(anim_data):
    if anim_data is None:
        return None
    if getattr(anim_data, "action_slot", None) is not None:
        return anim_data.action_slot
    slots = getattr(anim_data, "action_slots", None)
    if slots is not None:
        return getattr(slots, "active", None)
    return None


def _get_channelbag(action, anim_data=None, ensure=False):
    if action is None:
        return None
    slot = _get_action_slot(anim_data) if anim_data else None
    try:
        if ensure and hasattr(bpy_extras.anim_utils, "action_ensure_channelbag_for_slot"):
            return bpy_extras.anim_utils.action_ensure_channelbag_for_slot(action, slot)
        if not ensure and hasattr(bpy_extras.anim_utils, "action_get_channelbag_for_slot"):
            return bpy_extras.anim_utils.action_get_channelbag_for_slot(action, slot)
    except Exception:
        pass
    return None


def _get_action_fcurves(action_ref, anim_data=None):
    """
    Read-only access to fcurves for a given action/slot.
    We do NOT "ensure" channelbags here to avoid creating empty ones.
    """
    action = _unwrap_action(action_ref)
    channelbag = _get_channelbag(action, anim_data, ensure=False)
    if channelbag and hasattr(channelbag, "fcurves"):
        return channelbag.fcurves
    return getattr(action, "fcurves", None)


# ------------------------------------------------------------------------
#  Small utilities
# ------------------------------------------------------------------------

def cursor(cursor_mode):
    def cursor_decorator(func):
        def wrapper(self, context, *args, **kwargs):
            context.window.cursor_modal_set(cursor_mode)
            try:
                return func(self, context, *args, **kwargs)
            finally:
                context.window.cursor_modal_restore()

        return wrapper

    return cursor_decorator


def bone_name(prefix, position, side, index=0):
    if index == 0:
        return '%s.%s.%s' % (prefix, position, side)
    else:
        return '%s.%s.%s.%03d' % (prefix, position, side, index)


def bone_range(bones, name_prefix, position, side):
    for index in itertools.count():
        name = bone_name(name_prefix, position, side, index)
        if name in bones:
            yield bones[name]
        else:
            break


def find_wheelbrake_bone(bones, position, side, index):
    other_side = 'R' if side == 'L' else 'L'
    name_prefix = 'WheelBrake'
    bone = bones.get(bone_name(name_prefix, position, side, index))
    if bone:
        return bone
    bone = bones.get(bone_name(name_prefix, position, other_side, index))
    if bone:
        return bone
    if index > 0:
        bone = bones.get(bone_name(name_prefix, position, side))
        if bone:
            return bone
        bone = bones.get(bone_name(name_prefix, position, other_side))
        if bone:
            return bone
    backward_compatible_bone_name = '%s Wheels' % ('Front' if position == 'Ft' else 'Back')
    return bones.get(backward_compatible_bone_name)


def clear_property_animation(context, property_name, remove_keyframes=True):
    """
    Remove the FCurve for a given custom property (if present)
    and reset the property to 0.0.
    """
    obj = context.object
    anim_data = obj.animation_data

    if remove_keyframes and anim_data and anim_data.action:
        fcurve_datapath = '["%s"]' % property_name
        action = _unwrap_action(anim_data.action)
        channelbag = _get_channelbag(action, anim_data, ensure=False)
        if channelbag and hasattr(channelbag, "fcurves"):
            fcurve = channelbag.fcurves.find(fcurve_datapath, index=0)
            if fcurve is not None:
                channelbag.fcurves.remove(fcurve)
        else:
            fcurves = _get_action_fcurves(anim_data.action, anim_data)
            if fcurves is not None:
                fcurve = fcurves.find(fcurve_datapath)
                if fcurve is not None:
                    fcurves.remove(fcurve)

    obj[property_name] = 0.0


def create_property_animation(context, property_name):
    """
    Ensure there is an FCurve for a given custom property and return it.
    """
    obj = context.object
    if obj.animation_data is None:
        obj.animation_data_create()
    anim_data = obj.animation_data
    if anim_data.action is None:
        anim_data.action = bpy.data.actions.new("%sAction" % obj.name)

    action = _unwrap_action(anim_data.action)
    channelbag = _get_channelbag(action, anim_data, ensure=True)
    fcurve_datapath = '["%s"]' % property_name

    if channelbag and hasattr(channelbag, "fcurves"):
        return channelbag.fcurves.ensure(
            data_path=fcurve_datapath,
            index=0,
            group_name='Wheels rotation'
        )

    fcurves = _get_action_fcurves(anim_data.action, anim_data)
    if fcurves:
        existing = fcurves.find(fcurve_datapath)
        return existing or fcurves.new(
            fcurve_datapath,
            index=0,
            action_group='Wheels rotation'
        )
    return None


def fix_old_steering_rotation(rig_object):
    """
    Fix armature generated with Rigacar version < 6.0
    """
    if rig_object.pose and rig_object.pose.bones:
        if 'MCH-Steering.rotation' in rig_object.pose.bones:
            rig_object.pose.bones['MCH-Steering.rotation'].rotation_mode = 'QUATERNION'


def _guess_root_bone_name(obj: bpy.types.Object) -> str:
    """
    Try to find the main root/body bone to measure car motion from.
    """
    # Common names first
    for name in ("Root", "root", "ROOT", "Body", "body", "ROOT_M"):
        if name in obj.pose.bones:
            return name

    # Fallback: first bone without parent
    for pb in obj.pose.bones:
        if pb.parent is None:
            return pb.name

    # Last resort
    return next(iter(obj.pose.bones.keys()))


# ------------------------------------------------------------------------
#  Base class for baking operators
# ------------------------------------------------------------------------

class BakingOperator(object):
    frame_start: bpy.props.IntProperty(name='Start Frame', min=1)
    frame_end: bpy.props.IntProperty(name='End Frame', min=1)
    keyframe_tolerance: bpy.props.FloatProperty(name='Keyframe tolerance', min=0, default=.01)

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and
            context.object.data is not None and
            'Car Rig' in context.object.data and
            context.object.data['Car Rig'] and
            context.object.mode in {'POSE', 'OBJECT'}
        )

    def invoke(self, context, event):
        obj = context.object
        if obj.animation_data is None:
            obj.animation_data_create()
        if obj.animation_data.action is None:
            obj.animation_data.action = bpy.data.actions.new("%sAction" % obj.name)

        action = obj.animation_data.action
        action_start = int(action.frame_range[0])
        action_end = int(action.frame_range[1])

        if action_start == action_end:
            self.frame_start = context.scene.frame_start
            self.frame_end = context.scene.frame_end
        else:
            self.frame_start = action_start
            self.frame_end = action_end

        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.prop(self, 'frame_start')
        self.layout.prop(self, 'frame_end')
        self.layout.prop(self, 'keyframe_tolerance')


# ------------------------------------------------------------------------
#  Wheel rotation bake operator
# ------------------------------------------------------------------------

class ANIM_OT_carWheelsRotationBake(bpy.types.Operator, BakingOperator):
    bl_idname = 'anim.car_wheels_rotation_bake'
    bl_label = 'Bake wheels rotation'
    bl_description = 'Automatically generates wheels animation based on Root bone animation.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.object['wheels_on_y_axis'] = False
        self._bake_wheels_rotation(context)
        return {'FINISHED'}

    @cursor('WAIT')
    def _bake_wheels_rotation(self, context):
        bones = context.object.data.bones

        wheel_bones = []
        brake_bones = []
        for position, side in itertools.product(('Ft', 'Bk'), ('L', 'R')):
            for index, wheel_bone in enumerate(
                bone_range(bones, 'MCH-Wheel.rotation', position, side)
            ):
                wheel_bones.append(wheel_bone)
                brake_bones.append(
                    find_wheelbrake_bone(bones, position, side, index) or wheel_bone
                )

        # Clear old wheel property animations
        for property_name in map(lambda wheel_bone: wheel_bone.name.replace('MCH-', ''), wheel_bones):
            clear_property_animation(context, property_name)

        # Bake each wheel property from evaluated motion
        for wheel_bone, brake_bone in zip(wheel_bones, brake_bones):
            self._bake_wheel_rotation(context, wheel_bone, brake_bone)

    def _evaluate_distance_per_frame(self, context, bone, brake_bone):
        """
        Compute travel distance per frame based on the *armature object's*
        world-space motion, then convert to wheel rotation.

        This covers the common case where the path/constraint is on the
        armature object instead of a specific root bone.
        """
        scene = context.scene
        obj = context.object
        depsgraph = context.evaluated_depsgraph_get()

        # Wheel radius from the wheel MCH bone length
        radius = bone.length * 0.5 if bone.length > 0.0 else 1.0

        # Assume car forward is +Y in object local space
        forward_local = mathutils.Vector((0.0, 1.0, 0.0))

        def sample(frame):
            scene.frame_set(frame)
            eval_obj = obj.evaluated_get(depsgraph)

            # Object world transform
            mat_obj = eval_obj.matrix_world
            pos = mat_obj.translation.copy()
            rot_q = mat_obj.to_quaternion()

            # Brake from brake bone (pose space)
            brake_pb = eval_obj.pose.bones[brake_bone.name]
            brake_y = brake_pb.scale.y

            return pos, rot_q, brake_y

        # Initial state
        prev_pos, prev_rot_q, prev_brake_y = sample(self.frame_start)
        distance = 0.0
        prev_speed = 0.0

        # Always put a key on the first frame
        yield self.frame_start, distance

        for f in range(self.frame_start + 1, self.frame_end + 1):
            pos, rot_q, brake_y = sample(f)

            # Linear displacement this frame
            speed_vec = pos - prev_pos

            # Apply brake factor as original (2 * scale_y - 1)
            speed_vec *= (2.0 * brake_y - 1.0)

            # Signed speed along car forward direction
            forward_world = rot_q @ forward_local
            speed = speed_vec.length
            if speed != 0.0:
                speed = math.copysign(speed, forward_world.dot(speed_vec))

            # Convert linear distance to wheel angle (radians)
            angular_speed = -speed / (2.0 * math.pi * radius)

            drop_keyframe = False
            if angular_speed == 0.0:
                drop_keyframe = (prev_speed == angular_speed)
            elif prev_speed != 0.0:
                drop_keyframe = abs(1.0 - prev_speed / angular_speed) < self.keyframe_tolerance / 10.0

            if not drop_keyframe:
                prev_speed = angular_speed
                # Key on previous frame (matches original logic)
                yield f - 1, distance

            distance += angular_speed
            prev_pos = pos

        # Final frame
        yield self.frame_end, distance

    def _bake_wheel_rotation(self, context, bone, brake_bone):
        """
        Create/update the custom property FCurve on the original action,
        based on evaluated rig motion.
        """
        # Ensure FCurve exists for this wheel property
        prop_name = bone.name.replace('MCH-', '')
        fc_rot = create_property_animation(context, prop_name)
        if fc_rot is None:
            return

        # Reset wheel bone local transform to avoid interference
        pb: bpy.types.PoseBone = context.object.pose.bones[bone.name]
        pb.matrix_basis.identity()

        # Evaluate distance (integrated angle) and key it
        for f, distance in self._evaluate_distance_per_frame(context, bone, brake_bone):
            kf = fc_rot.keyframe_points.insert(f, distance)
            kf.interpolation = 'LINEAR'
            kf.type = 'JITTER'


# ------------------------------------------------------------------------
#  Steering bake operator
# ------------------------------------------------------------------------

class ANIM_OT_carSteeringBake(bpy.types.Operator, BakingOperator):
    bl_idname = 'anim.car_steering_bake'
    bl_label = 'Bake car steering'
    bl_description = 'Automatically generates steering animation based on Root bone animation.'
    bl_options = {'REGISTER', 'UNDO'}

    rotation_factor: bpy.props.FloatProperty(
        name='Rotation factor',
        min=.1,
        default=1.0
    )

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.prop(self, 'frame_start')
        self.layout.prop(self, 'frame_end')
        self.layout.prop(self, 'rotation_factor')
        self.layout.prop(self, 'keyframe_tolerance')

    def execute(self, context):
        if self.frame_end > self.frame_start:
            bones = context.object.data.bones
            if 'Steering' in bones and 'MCH-Steering.rotation' in bones:
                steering = bones['Steering']
                mch_steering_rotation = bones['MCH-Steering.rotation']
                bone_offset = abs(steering.head_local.y - mch_steering_rotation.head_local.y)
                self._bake_steering_rotation(context, bone_offset, mch_steering_rotation)
        return {'FINISHED'}

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi] for stable deltas."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _evaluate_rotation_per_frame(self, context, bone_offset, bone):
        """
        Compute steering per frame based on the *rate of change* of the
        armature object's heading, and output a value for EVERY frame.

        This:
        - goes back to ~0 on straight sections (heading stops changing),
        - is smooth (we use central differences, no keyframe decimation),
        - lets the FCurve interpolation handle the 'lerp' between directions.
        """
        scene = context.scene
        obj = context.object
        depsgraph = context.evaluated_depsgraph_get()

        # Car forward axis in object local space
        forward_local = mathutils.Vector((0.0, 1.0, 0.0))

        def heading_at(frame):
            scene.frame_set(frame)
            eval_obj = obj.evaluated_get(depsgraph)
            mat_obj = eval_obj.matrix_world
            forward_world = mat_obj.to_quaternion() @ forward_local
            # Heading in ground plane (X–Y)
            return math.atan2(forward_world.x, forward_world.y)

        # Pre-sample headings for a small neighborhood around the range
        headings = {}
        for f in range(self.frame_start - 1, self.frame_end + 2):
            headings[f] = heading_at(f)

        # Central difference per-frame heading change
        for f in range(self.frame_start, self.frame_end):
            h_prev = headings.get(f - 1, headings[f])
            h_next = headings.get(f + 1, headings[f])

            # Central difference (note the 0.5 factor)
            delta_heading = self._wrap_angle(h_next - h_prev) * 0.5

            # Steering proportional to turning rate.
            # Flip the sign if the direction is reversed in your rig.
            steering_position = -delta_heading * self.rotation_factor * 180

            yield f, steering_position

    @cursor('WAIT')
    def _bake_steering_rotation(self, context, bone_offset, bone):
        clear_property_animation(context, 'Steering.rotation')
        fix_old_steering_rotation(context.object)
        fc_rot = create_property_animation(context, 'Steering.rotation')
        if fc_rot is None:
            return

        # Reset steering bone local transform
        pb: bpy.types.PoseBone = context.object.pose.bones[bone.name]
        pb.matrix_basis.identity()

        # Evaluate steering directly from evaluated rig
        for f, steering_pos in self._evaluate_rotation_per_frame(context, bone_offset, bone):
            kf = fc_rot.keyframe_points.insert(f, steering_pos)
            kf.type = 'KEYFRAME'
            kf.interpolation = 'BEZIER'


# ------------------------------------------------------------------------
#  Clear baked animation operator
# ------------------------------------------------------------------------

class ANIM_OT_carClearSteeringWheelsRotation(bpy.types.Operator):
    bl_idname = "anim.car_clear_steering_wheels_rotation"
    bl_label = "Clear baked animation"
    bl_description = "Clear generated rotation for steering and wheels"
    bl_options = {'REGISTER', 'UNDO'}

    clear_steering: bpy.props.BoolProperty(
        name="Steering",
        description="Clear generated animation for steering",
        default=True
    )
    clear_wheels: bpy.props.BoolProperty(
        name="Wheels",
        description="Clear generated animation for wheels",
        default=True
    )

    def draw(self, context):
        self.layout.use_property_decorate = False
        self.layout.label(text='Clear generated keyframes for')
        self.layout.prop(self, property='clear_steering')
        self.layout.prop(self, property='clear_wheels')

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and
            context.object.data is not None and
            context.object.data.get('Car Rig')
        )

    def execute(self, context):
        re_wheel_propname = re.compile(r'^Wheel\.rotation\.(Ft|Bk)\.[LR](\.\d+)?$')
        for prop in context.object.keys():
            if prop == 'Steering.rotation':
                clear_property_animation(context, prop, remove_keyframes=self.clear_steering)
            elif re_wheel_propname.match(prop):
                clear_property_animation(context, prop, remove_keyframes=self.clear_wheels)

        # Hack to force Blender to refresh custom property drivers etc.
        mode = context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT' if mode == 'POSE' else 'POSE')
        bpy.ops.object.mode_set(mode=mode)
        return {'FINISHED'}


# ------------------------------------------------------------------------
#  Registration
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(ANIM_OT_carWheelsRotationBake)
    bpy.utils.register_class(ANIM_OT_carSteeringBake)
    bpy.utils.register_class(ANIM_OT_carClearSteeringWheelsRotation)


def unregister():
    bpy.utils.unregister_class(ANIM_OT_carClearSteeringWheelsRotation)
    bpy.utils.unregister_class(ANIM_OT_carSteeringBake)
    bpy.utils.unregister_class(ANIM_OT_carWheelsRotationBake)


if __name__ == "__main__":
    register()
