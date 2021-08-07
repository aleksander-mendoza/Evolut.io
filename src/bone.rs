use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    particle_ids:glm::UVec4,
    center:glm::Vec3,
    color:f32,
}

impl Bone{
    pub fn new(particle_ids:glm::UVec4) -> Self{
        Self{
            particle_ids,
            center:glm::vec3(0.,0.,0.),
            color:rand::random()
        }
    }
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
                offset: offset_of!(Self, center) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  u32::FORMAT,
                offset: offset_of!(Self, color) as u32,
            }
        ]
    }
}