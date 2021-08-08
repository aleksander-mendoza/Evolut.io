use ash::vk;
use ash::vk::Format;

pub trait VertexSource: Copy{
    fn get_attribute_descriptions(binding:u32) -> Vec<vk::VertexInputAttributeDescription>;
}

pub trait VertexAttrib {
    const FORMAT:vk::Format;
}

impl VertexAttrib for glm::Vec2{
    const FORMAT: Format = vk::Format::R32G32_SFLOAT;
}

impl VertexAttrib for glm::Vec3{
    const FORMAT: Format = vk::Format::R32G32B32_SFLOAT;
}

impl VertexAttrib for glm::Vec4{
    const FORMAT: Format = vk::Format::R32G32B32A32_SFLOAT;
}

impl VertexAttrib for glm::UVec2{
    const FORMAT: Format = vk::Format::R32G32_UINT;
}

impl VertexAttrib for glm::UVec3{
    const FORMAT: Format = vk::Format::R32G32B32_UINT;
}

impl VertexAttrib for glm::UVec4{
    const FORMAT: Format = vk::Format::R32G32B32A32_UINT;
}
impl VertexAttrib for u8{
    const FORMAT: Format = vk::Format::R8_UINT;
}
impl VertexAttrib for u16 {
    const FORMAT: Format = vk::Format::R16_UINT;
}
impl VertexAttrib for u32 {
    const FORMAT: Format = vk::Format::R32_UINT;
}
impl VertexAttrib for f32 {
    const FORMAT: Format = vk::Format::R32_SFLOAT;
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, packed)]
pub struct u8_u8_u8_u8 {
    pub d0: u8,
    pub d1: u8,
    pub d2: u8,
    pub d3: u8,
}

impl u8_u8_u8_u8 {
    pub fn as_u32(&self) -> &u32 {
        unsafe { std::mem::transmute::<&u8_u8_u8_u8, &u32>(self) }
    }
    pub fn new(d0: u8, d1: u8, d2: u8, d3: u8) -> u8_u8_u8_u8 {
        u8_u8_u8_u8 { d0, d1, d2, d3 }
    }
}

impl VertexAttrib for u8_u8_u8_u8 {
    const FORMAT: Format = vk::Format::R8G8B8A8_UINT;
}

impl From<(u8, u8, u8, u8)> for u8_u8_u8_u8 {
    fn from(other: (u8, u8, u8, u8)) -> Self {
        u8_u8_u8_u8::new(other.0, other.1, other.2, other.3)
    }
}

impl From<&[u8; 4]> for u8_u8_u8_u8 {
    fn from(other: &[u8; 4]) -> Self {
        u8_u8_u8_u8::new(other[0], other[1], other[2], other[3])
    }
}

impl From<u32> for u8_u8_u8_u8 {
    fn from(other: u32) -> Self {
        unsafe { std::mem::transmute::<u32, u8_u8_u8_u8>(other) }
    }
}



#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, packed)]
pub struct u8_u8 {
    pub d0: u8,
    pub d1: u8,
}

impl u8_u8 {
    pub fn new(d0: u8, d1: u8) -> u8_u8 {
        u8_u8 { d0, d1 }
    }
}

impl VertexAttrib for u8_u8 {
    const FORMAT: Format = vk::Format::R8G8_UINT;
}

impl From<(u8, u8)> for u8_u8 {
    fn from(other: (u8, u8)) -> Self {
        u8_u8::new(other.0, other.1)
    }
}

impl From<&[u8; 2]> for u8_u8 {
    fn from(other: &[u8; 2]) -> Self {
        u8_u8::new(other[0], other[1])
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
