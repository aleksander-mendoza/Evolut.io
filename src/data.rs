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

impl Attribute for glm::Vec4{
    const FORMAT: Format = vk::Format::R32G32B32A32_SFLOAT;
}

impl Attribute for u8{
    const FORMAT: Format = vk::Format::R8_UINT;
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct VertexClr {
    pub pos:glm::Vec2,
    pub color:glm::Vec3
}

impl VertexSource for VertexClr {
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


impl VertexSource for glm::Mat4 {
    fn get_attribute_descriptions(binding:u32) -> Vec<vk::VertexInputAttributeDescription>{
        (0..4).into_iter().map(|location|vk::VertexInputAttributeDescription {
            binding,
            location,
            format:  glm::Vec4::FORMAT,
            offset: location*std::mem::size_of::<glm::Vec4>() as u32,
        }).collect()
    }
}


impl VertexSource for u8 {
    fn get_attribute_descriptions(binding:u32) -> Vec<vk::VertexInputAttributeDescription>{
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format: u8::FORMAT,
                offset: 0,
            },
        ]
    }
}


#[repr(C)]
#[derive(Debug, Clone)]
pub struct VertexTex {
    pub pos:glm::Vec2,
    pub tex:glm::Vec2,
}

impl VertexSource for VertexTex {
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
                format: glm::Vec2::FORMAT,
                offset: offset_of!(Self, tex) as u32,
            },
        ]
    }
}


#[repr(C)]
#[derive(Debug, Clone)]
pub struct VertexClrTex {
    pub pos:glm::Vec2,
    pub clr:glm::Vec3,
    pub tex:glm::Vec2,
}

impl VertexSource for VertexClrTex {
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
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, clr) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format: glm::Vec2::FORMAT,
                offset: offset_of!(Self, tex) as u32,
            },
        ]
    }
}
