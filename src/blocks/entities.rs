use crate::render::data::u8_u8;

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct BoneInstance {
    position: glm::Vec3,
    body_part_and_bone_variant: u8_u8,
    rotation: glm::Quat, //rotation
}

impl BoneInstance {
    pub fn new(entity_position: &glm::Vec3, entity_rotation: &glm::Quat, body_part: BodyPart, bone_variant: u8, mut rotation: glm::Quat) -> Self {
        assert_eq!(std::mem::size_of::<BodyPart>(), std::mem::size_of::<u8>());
        let position = body_part.absolute_rotated_position(entity_position,entity_rotation);
        rotation = entity_rotation * rotation;
        Self { position, body_part_and_bone_variant: u8_u8::new(body_part as u8, bone_variant), rotation }
    }
    pub fn body_part(&self) -> BodyPart {
        self.body_part_and_bone_variant.d0 as BodyPart
    }
    pub fn bone_variant(&self) -> u8 {
        self.body_part_and_bone_variant.d1
    }

    pub fn update(&mut self,entity_position: &glm::Vec3, entity_rotation:&glm::Quat){
        let body_part = self.body_part();
        self.position=body_part.absolute_rotated_position(entity_position,entity_rotation);
        self.rotation = entity_rotation.clone();
    }
}
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum BodyPart {
    ZombieLeftLeg = 0,
    ZombieRightLeg = 1,
    ZombieTorso = 2,
    ZombieHead = 3,
    ZombieLeftArm = 4,
    ZombieRightArm = 5,
}

/**Size of a single pixel in world-space. Size of a block is 1x1x1. Every block is composed of 16x16 pixels.
 Hence size of a pixel is 1/16. */
pub const U: f32 = 1. / 16.;

impl BodyPart {
    /**By default every bone is centered according to it's joint position (joint is the center of bone
    and rotation is performed about that joint. Afterwards, the bone is translated into the right position,
    relative to entity's world-space position. Usually mobs position is some point on the ground, right in the
    middle of mob's body mass). This function returns the vector, by which joint positions must be translated in order
    to place the bone in the right place (e.g. head is on top ob torso, legs are below torso,
    torso is several pixels above the entity's absolute position)*/
    pub fn relative_position(&self) -> glm::Vec3 {
        match self {
            BodyPart::ZombieLeftLeg => glm::vec3(U * 0., U * 12., U * 0.),
            BodyPart::ZombieRightLeg => glm::vec3(U * 0., U * 12., U * 0.),
            BodyPart::ZombieTorso => glm::vec3(U * 0., U * 18., U * 0.),
            BodyPart::ZombieHead => glm::vec3(U * 0., U * 24., U * 0.),
            BodyPart::ZombieLeftArm => glm::vec3(U * (-4.), U * 22., U * 0.),
            BodyPart::ZombieRightArm => glm::vec3(U * (4.), U * 22., U * 0.),
        }
    }

    pub fn absolute_position(&self, entity_abs_position: &glm::Vec3) -> glm::Vec3 {
        self.relative_position() + entity_abs_position
    }

    pub fn absolute_rotated_position(&self, entity_abs_position: &glm::Vec3, entity_rotation: &glm::Quat) -> glm::Vec3 {
        glm::quat_rotate_vec3(&entity_rotation, &self.relative_position()) + entity_abs_position
    }
}


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ZombieVariant {
    Zombie = 0,
    Steve = 1,
}

pub enum Entity {
    Zombie(ZombieVariant)
}

impl Entity {
    pub fn len(&self) -> usize {
        match self {
            Entity::Zombie(_) => 6
        }
    }
}

enum EntityBones {
    Zombie(/*left leg*/usize, /*right leg*/usize, /*torso*/usize, /*head*/usize, /*left arm*/usize, /*right arm*/usize)
}

pub struct Entities {
    bones: Vec<BoneInstance>,
    bone_to_entity: Vec<usize>,
    entity_to_bones: Vec<EntityBones>,
}

impl Entities {
    pub fn new() -> Self {
        Self { bones: vec![], bone_to_entity: vec![], entity_to_bones: vec![] }
    }
    pub fn bone_slice(&self) -> &[BoneInstance] {
        &self.bones
    }
    fn add_bone(&mut self, owning_entity: usize, bone: BoneInstance) -> usize {
        let new_idx = self.bones.len();
        assert_eq!(self.bones.len(), self.bone_to_entity.len());
        self.bones.push(bone);
        self.bone_to_entity.push(owning_entity);
        new_idx
    }
    pub fn swap_remove_bone(&mut self, bone_idx: usize) {
        assert_eq!(self.bone_to_entity.len(), self.bones.len());
        self.bones.swap_remove(bone_idx);
        let &corrupt_owner = self.bone_to_entity.last().unwrap();
        let owner_of_deleted_bone = self.bone_to_entity[bone_idx];
        let misplaced_bone = &self.bones[bone_idx];
        if owner_of_deleted_bone != corrupt_owner {
            let body_part = misplaced_bone.body_part();
            match &mut self.entity_to_bones[corrupt_owner] {
                EntityBones::Zombie(ll, rl, t, h, la, ra) => match body_part {
                    BodyPart::ZombieLeftLeg => { *ll = bone_idx }
                    BodyPart::ZombieRightLeg => { *rl = bone_idx }
                    BodyPart::ZombieTorso => { *t = bone_idx }
                    BodyPart::ZombieHead => { *h = bone_idx }
                    BodyPart::ZombieLeftArm => { *la = bone_idx }
                    BodyPart::ZombieRightArm => { *ra = bone_idx }
                }
            }
        }
        self.bone_to_entity.swap_remove(bone_idx);
        assert_eq!(corrupt_owner,self.bone_to_entity[bone_idx]);
        assert_eq!(self.bone_to_entity.len(), self.bones.len());
    }
    pub fn update(&mut self, entity_id: usize, entity_position: &glm::Vec3, entity_rotation:&glm::Quat) {
        match self.entity_to_bones[entity_id]{
            EntityBones::Zombie(left_leg, right_leg, torso, head, left_arm, right_arm) => {
                self.bones[left_leg].update(entity_position, entity_rotation);
                self.bones[right_leg].update(entity_position, entity_rotation);
                self.bones[torso].update(entity_position, entity_rotation);
                self.bones[head].update(entity_position, entity_rotation);
                self.bones[left_arm].update(entity_position, entity_rotation);
                self.bones[right_arm].update(entity_position, entity_rotation);
            }
        }
    }
    pub fn push(&mut self, ent: Entity, entity_position: &glm::Vec3, entity_rotation:&glm::Quat) -> usize {
        assert_eq!(self.bone_to_entity.len(), self.bones.len());
        match ent {
            Entity::Zombie(variant) => {
                let entity_id = self.entity_to_bones.len();
                let variant = variant as u8;
                let bones = EntityBones::Zombie(
                    self.add_bone(entity_id, BoneInstance::new(entity_position, entity_rotation,BodyPart::ZombieLeftLeg, variant, glm::quat_identity())),
                    self.add_bone(entity_id, BoneInstance::new(entity_position, entity_rotation,BodyPart::ZombieRightLeg, variant, glm::quat_identity())),
                    self.add_bone(entity_id, BoneInstance::new(entity_position, entity_rotation,BodyPart::ZombieTorso, variant, glm::quat_identity())),
                    self.add_bone(entity_id, BoneInstance::new(entity_position, entity_rotation,BodyPart::ZombieHead, variant, glm::quat_identity())),
                    self.add_bone(entity_id, BoneInstance::new(entity_position, entity_rotation,BodyPart::ZombieLeftArm, variant, glm::quat_identity())),
                    self.add_bone(entity_id, BoneInstance::new(entity_position, entity_rotation,BodyPart::ZombieRightArm, variant, glm::quat_identity())),
                );
                self.entity_to_bones.push(bones);
                entity_id
            }
        }
    }

    pub fn remove(&mut self, entity_id: usize) {
        let bones_to_remove = &self.entity_to_bones[entity_id];
        match bones_to_remove {
            &EntityBones::Zombie(left_leg, right_leg, torso, head, left_arm, right_arm) => {
                self.swap_remove_bone(left_leg);
                self.swap_remove_bone(right_leg);
                self.swap_remove_bone(torso);
                self.swap_remove_bone(head);
                self.swap_remove_bone(left_arm);
                self.swap_remove_bone(right_arm);
            }
        }
        self.entity_to_bones.swap_remove(entity_id);
        assert_eq!(self.bone_to_entity.len(),self.bones.len())
    }
}
