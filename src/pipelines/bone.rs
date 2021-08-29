use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    center:glm::Vec3,
    color1:f32,
    size:glm::Vec3,
    color2:f32,
}

impl Bone{
    pub fn new(center:glm::Vec3,
               color1:f32,
               size:glm::Vec3,
               color2:f32) -> Self{
        Self{
            center,
            color1,
            size,
            color2,
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
                offset: offset_of!(Self, color1)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, size)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 3,
                format:  f32::FORMAT,
                offset: offset_of!(Self, color2) as u32,
            },

        ]
    }
}