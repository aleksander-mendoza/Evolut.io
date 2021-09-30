use crate::render::data::{u8_u8_u8_u8, VertexAttrib, u16_u16, u8_u8_u16};
use crate::blocks::block::Block;
use crate::blocks::face_orientation::FaceOrientation;
use crate::blocks::world_size::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH};
use crate::render::data::VertexSource;
use ash::vk;
use crate::blocks::BlockId;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C, packed)]
pub struct Face {
    coords: u8_u8_u8_u8,
    tex_id: u8_u8_u16
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C, packed)]
pub struct FaceRelocation{
    dst_offset:u32,
    face:Face,
}

impl VertexSource for Face {
    fn get_attribute_descriptions(binding:u32) -> Vec<vk::VertexInputAttributeDescription>{
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  u8_u8_u8_u8::FORMAT,
                offset: offset_of!(Self, coords) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format: u8_u8_u8_u8::FORMAT, // There is no such thing as u8_u8_u16::FORMAT.
                // We have to choose either u16_u16 or u8_u8_u8_u8. In shaders it's somewhat easier to work with the latter.
                offset: offset_of!(Self, tex_id) as u32,
            },
        ]
    }
}


impl Face {
    pub fn as_u32(&self)->&u32{
        self.coords.as_u32()
    }
    pub fn as_mut_u32(&mut self)->&mut u32{
        self.coords.as_mut_u32()
    }
    pub fn zero()->Self{
        Face{ coords: u8_u8_u8_u8::new(0,0,0,0), tex_id: u8_u8_u16::new(0,0,0) }
    }
    pub fn update_texture(&mut self, new_block: BlockId) {
        let ort = self.block_orientation();
        self.tex_id.d2 = new_block.texture_id(ort) as u16;
    }
    pub fn coords_and_ort(&self) -> u32 {
        self.coords.as_u32().clone()
    }
    pub fn x(&self) -> u8 {
        self.coords.d0
    }
    pub fn matches_coords(&self, x: u8, y: u8, z: u8) -> bool {
        self.x() == x && self.y() == y && self.z() == z
    }
    pub fn matches_block_coords(&self, x: usize, y: usize, z: usize) -> bool {
        self.block_x() == x && self.block_y() == y && self.block_z() == z
    }
    pub fn y(&self) -> u8 {
        self.coords.d1
    }
    pub fn z(&self) -> u8 {
        self.coords.d2
    }
    pub fn orientation(&self) -> u8 {
        self.coords.d3
    }
    pub fn block_x(&self) -> usize {
        self.coords.d0 as usize
    }
    pub fn block_y(&self) -> usize {
        self.coords.d1 as usize
    }
    pub fn block_z(&self) -> usize {
        self.coords.d2 as usize
    }
    pub fn block_orientation(&self) -> FaceOrientation {
        FaceOrientation::from(self.coords.d3)
    }
    pub fn texture_id(&self) -> u16 {
        self.tex_id.d2
    }
    pub fn encode_coords_and_ort(x: u8, y: u8, z: u8, orientation: FaceOrientation) -> u32 {
        assert!((x as usize) < CHUNK_WIDTH);
        assert!((y as usize) < CHUNK_HEIGHT);
        assert!((z as usize) < CHUNK_DEPTH);
        u8_u8_u8_u8::from((x, y, z, orientation as u8)).as_u32().clone()
    }
    pub fn from_coords_and_ort(chunk_x:u8, chunk_y:u8,x: u8, y: u8, z: u8, orientation: FaceOrientation, texture_id: u16) -> Self {
        assert!((x as usize) < CHUNK_WIDTH, "{} < {}", x ,CHUNK_WIDTH);
        assert!((y as usize) < CHUNK_HEIGHT, "{} < {}", y ,CHUNK_HEIGHT);
        assert!((z as usize) < CHUNK_DEPTH, "{} < {}", z ,CHUNK_DEPTH);
        assert_eq!(
            std::mem::size_of::<FaceOrientation>(),
            std::mem::size_of::<u8>()
        );
        Self { coords: u8_u8_u8_u8::from((x, y, z, orientation as u8)), tex_id:  u8_u8_u16::new(chunk_x,chunk_y,texture_id) }
    }
}