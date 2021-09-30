use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;
use crate::neat::num::Num;
use crate::blocks::block_properties::DIRT;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    new_center: glm::Vec3,
    texture_for_block_id: u32,
    impulse: glm::Vec3,
    mass: f32 ,
    old_center: glm::Vec3,
    entity_idx: u32,
    position_relative_to_parent: glm::Vec3,
    parent_bone_idx: u32,
    half_side_length: f32,//width == 2*half_side_length && depth == 2*half_side_length
    half_height: f32,
    yaw_and_pitch: glm::Vec2, // yaw = how high (up/down) is the entity looking
    //  pitch = which direction (left/right) is the entity looking
}

impl Bone{
    pub fn new(center:glm::Vec3,
               half_side_length:f32,
               height:f32,
               mass:f32) -> Self{
        let velocity = f32::random_vec3()*0.02-glm::vec3(0.01,0.01,0.01);
        Self{
            new_center:center,
            half_side_length,
            half_height:height/2f32,
            old_center:center - velocity,
            entity_idx: 0,
            position_relative_to_parent: Default::default(),
            mass,
            yaw_and_pitch:f32::random_vec2(),
            texture_for_block_id:DIRT.id(),
            impulse: glm::vec3(0.,0.,0.),
            parent_bone_idx: u32::MAX
        }
    }
}

impl VertexSource for Bone{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, new_center) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format:  f32::FORMAT,
                offset: offset_of!(Self, half_side_length)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  u32::FORMAT,
                offset: offset_of!(Self, texture_for_block_id)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 3,
                format:  f32::FORMAT,
                offset: offset_of!(Self, half_height) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 4,
                format:  glm::Vec2::FORMAT,
                offset: offset_of!(Self, yaw_and_pitch) as u32,
            },
        ]
    }
}