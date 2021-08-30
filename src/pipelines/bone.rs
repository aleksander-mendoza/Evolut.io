use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;
use ash::vk;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Bone{
    center:glm::Vec3,
    width:f32,
    color:glm::Vec3,
    depth:f32,
    direction:glm::Vec3,
    height:f32,
    velocity:glm::Vec3,
    mass:f32,
}

impl Bone{
    pub fn new(center:glm::Vec3,
               size:glm::Vec3,
               direction:glm::Vec3,
               color:glm::Vec3,
               mass:f32) -> Self{
        assert!(size.z>=size.x);
        assert!(2.*size.x>=size.z);
        let direction_xz_len =  (direction.x*direction.x + direction.z*direction.z).sqrt();
        Self{
            center,
            color,
            width:size.x,
            direction:direction / direction_xz_len, //normalize direction, but only the xz dimension has unit length
            height:size.y,
            velocity:glm::zero(),
            depth:size.z,
            mass
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
                offset: offset_of!(Self, width)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, color)  as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 3,
                format:  f32::FORMAT,
                offset: offset_of!(Self, depth) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 4,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, direction) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 5,
                format:  f32::FORMAT,
                offset: offset_of!(Self, height)  as u32,
            },
        ]
    }
}