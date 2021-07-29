use std::fmt::{Display, Formatter};
use crate::blocks::block_properties::{BLOCKS, STONE};
use crate::blocks::face_orientation::FaceOrientation;
use crate::render::data::{VertexSource, VertexAttrib};
use ash::vk::VertexInputAttributeDescription;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C, packed)]
pub struct Block {
    idx: u32,
}

impl VertexSource for Block{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            VertexInputAttributeDescription{
                location: 0,
                binding,
                format: u32::FORMAT,
                offset: offset_of!(Block, idx) as u32
            }
        ]
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Block {
    pub const fn air() -> Self {
        Self::new(0)
    }
    pub const fn new(idx: u32) -> Self {
        Self { idx }
    }
    pub fn weight(&self) -> u32 {
        (self.idx - 10).max(0)
    }
    pub fn is_solid(&self) -> bool {
        self.idx > 0
    }
    pub fn opacity(&self) -> f32 {
        BLOCKS[self.idx as usize].opacity()
    }
    pub fn is_air(&self) -> bool {
        self.idx == 0
    }
    pub fn texture_id(&self, ort: FaceOrientation) -> u32 {
        BLOCKS[self.idx as usize].get_texture_id(ort)
    }
    pub fn name(&self) -> &'static str {
        BLOCKS[self.idx as usize].name()
    }
    pub fn show_neighboring_faces(&self) -> bool { self.is_transparent() }
    pub fn show_my_faces(&self) -> bool { !self.is_air() }
    pub fn is_transparent(&self) -> bool{
        self.opacity() < 1.
    }
}
