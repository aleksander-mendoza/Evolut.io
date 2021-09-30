use std::fmt::{Display, Formatter};
use crate::blocks::block_properties::{BLOCKS, NO_OF_TRANSPARENT_BLOCKS};
use crate::blocks::face_orientation::FaceOrientation;
use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C, packed)]
pub struct Block {
    block_id:BlockId,
    block_id_copy:BlockId,
    humidity:u32,
    temperature:u32,
    new_humidity:u32,
    new_temperature:u32,
}
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C, packed)]
pub struct BlockId {
    id:u32
}
impl VertexSource for Block{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            VertexInputAttributeDescription{
                location: 0,
                binding,
                format: u32::FORMAT,
                offset: offset_of!(Block, block_id) as u32
            }
        ]
    }
}
impl VertexSource for BlockId{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            VertexInputAttributeDescription{
                location: 0,
                binding,
                format: u32::FORMAT,
                offset: offset_of!(BlockId, id) as u32
            }
        ]
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl BlockId {
    pub const fn air() -> Self {
        Self::new(0)
    }
    pub const fn new(id: u32) -> Self {
        Self { id }
    }
    pub fn weight(&self) -> u32 {
        (self.id - 10).max(0)
    }
    pub const fn id(&self) -> u32 {
        self.id
    }
    pub fn is_solid(&self) -> bool {
        self.id > NO_OF_TRANSPARENT_BLOCKS
    }
    pub const fn opacity(&self) -> f32 {
        BLOCKS[self.id as usize].opacity()
    }

    pub fn is_air(&self) -> bool {
        self.id == 0
    }
    pub fn texture_id(&self, ort: FaceOrientation) -> u32 {
        BLOCKS[self.id as usize].get_texture_id(ort)
    }
    pub fn name(&self) -> &'static str {
        BLOCKS[self.id as usize].name()
    }
    pub fn show_neighboring_faces(&self) -> bool { self.is_transparent() }
    pub fn show_my_faces(&self) -> bool { !self.is_air() }
    pub fn is_transparent(&self) -> bool{
        self.opacity() < 1.
    }
    pub fn is_opaque(&self) -> bool{
        self.opacity() == 1.
    }
}

