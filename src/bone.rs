use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    particle_ids:glm::UVec4,
    texture_variant:u32,
    part_variant:u32,
    dummy_field0:u32,
    dummy_field1:u32,
}


impl VertexSource for Bone{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  glm::UVec4::FORMAT,
                offset: offset_of!(Self, particle_ids) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format:  u32::FORMAT,
                offset: offset_of!(Self, texture_variant) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  f32::FORMAT,
                offset: offset_of!(Self, part_variant) as u32,
            }
        ]
    }
}