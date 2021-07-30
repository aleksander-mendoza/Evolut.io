use crate::render::data::VertexSource;
use ash::vk::VertexInputAttributeDescription;
use ash::vk;
use rand::random;
use crate::render::data::VertexAttrib;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Particle{
    pos:glm::Vec3,
    size:f32,
    color:glm::Vec4,
}
impl Particle{
    pub fn random()->Self{
        Self{
            pos: glm::vec3(random::<f32>()*32.,3.+random::<f32>()*8.,random::<f32>()*32.),
            size: random::<f32>()*100.,
            color: glm::vec4(random::<f32>(),random::<f32>(),random::<f32>(), 1.)
        }
    }
}
impl VertexSource for Particle{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format:  f32::FORMAT,
                offset: offset_of!(Self, size) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  glm::Vec4::FORMAT,
                offset: offset_of!(Self, color) as u32,
            }
        ]
    }
}