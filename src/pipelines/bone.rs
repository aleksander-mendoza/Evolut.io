use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    particle_ids:[u32;4],
    center:glm::Vec3,
    texture_variant:u32,
    normal:glm::Vec3,
    part_variant:u32,
}

impl Bone{
    pub fn new(particle_ids:[u32;4], part_variant:u32) -> Self{
        Self{
            particle_ids,
            center: glm::vec3(0.,0.,0.),
            texture_variant: 0,
            normal: glm::vec3(0.,0.,0.),
            part_variant
        }
    }
}

impl VertexSource for Bone{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  u32::FORMAT,
                offset: (offset_of!(Self, particle_ids) + std::mem::size_of::<u32>()*0) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format:  u32::FORMAT,
                offset: (offset_of!(Self, particle_ids) + std::mem::size_of::<u32>()*1) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  u32::FORMAT,
                offset: (offset_of!(Self, particle_ids) + std::mem::size_of::<u32>()*2) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 3,
                format:  u32::FORMAT,
                offset: (offset_of!(Self, particle_ids) + std::mem::size_of::<u32>()*3) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 4,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, center) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 5,
                format:  u32::FORMAT,
                offset: offset_of!(Self, texture_variant) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 6,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, normal) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 7,
                format:  u32::FORMAT,
                offset: offset_of!(Self, part_variant) as u32,
            },
        ]
    }
}