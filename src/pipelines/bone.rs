use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;
use crate::neat::num::Num;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    center:glm::Vec3,
    half_side_length:f32,
    direction:glm::Vec3,
    height:f32,
    impulse:glm::Vec3,
    mass:f32,
    velocity:glm::Vec3,
    entity_idx:u32,
    position_relative_to_parent:glm::Vec3,
    parent_bone_idx:u32,
    texture_coords:glm::Vec4,
}

impl Bone{
    pub fn new(center:glm::Vec3,
               half_side_length:f32,
               direction:glm::Vec3,
               height:f32,
               mass:f32) -> Self{
        let direction_xz_len =  (direction.x*direction.x + direction.z*direction.z).sqrt();
        Self{
            center,
            half_side_length,
            height,
            velocity:f32::random_vec3()*0.01,
            entity_idx: 0,
            position_relative_to_parent: Default::default(),
            mass,
            texture_coords:f32::random_vec4(),
            direction:direction / direction_xz_len, //normalize direction, but only the xz dimension has unit length
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
                offset: offset_of!(Self, center) as u32,
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
                format:  glm::Vec4::FORMAT,
                offset: offset_of!(Self, texture_coords)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 3,
                format:  f32::FORMAT,
                offset: offset_of!(Self, height) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 4,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, direction) as u32,
            },
        ]
    }
}