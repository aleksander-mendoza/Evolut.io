use ash::vk;
use ash::vk::Format;

pub trait VertexSource{
    fn get_attribute_descriptions(binding:u32) -> Vec<vk::VertexInputAttributeDescription>;
}

pub trait Attribute{
    const FORMAT:vk::Format;
}

impl Attribute for glm::Vec2{
    const FORMAT: Format = vk::Format::R32G32_SFLOAT;
}

impl Attribute for glm::Vec3{
    const FORMAT: Format = vk::Format::R32G32B32_SFLOAT;
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Vertex {
    pub pos:glm::Vec2,
    pub color:glm::Vec3
}

impl VertexSource for Vertex {
    fn get_attribute_descriptions(binding:u32) -> Vec<vk::VertexInputAttributeDescription>{
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  glm::Vec2::FORMAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format: glm::Vec3::FORMAT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}