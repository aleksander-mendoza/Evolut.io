use crate::blocks::chunk_faces::ChunkFaces;
use crate::blocks::world_size::WorldSize;
use crate::blocks::face_orientation::FaceOrientation;
use crate::blocks::block::Block;
use std::ops::{Deref, DerefMut};
use crate::render::device::Device;

pub struct WorldFaces {
    chunks: Vec<ChunkFaces>,
    size:WorldSize
}

impl WorldFaces{
    pub fn new(size:WorldSize, device:&Device)->Self{
        Self{ chunks: std::iter::repeat_with(||ChunkFaces::new(device)).take(size.total_chunks()).collect(), size }
    }
    pub fn size(&self)->&WorldSize{
        &self.size
    }
    pub(crate) fn remove_block_transparent(&mut self, x: usize, y: usize, z: usize) {
        self.get_chunk_mut(x, z).remove_block_transparent(x, y, z)
    }
    pub(crate) fn remove_block_opaque(&mut self, x: usize, y: usize, z: usize) {
        self.get_chunk_mut(x, z).remove_block_opaque(x, y, z)
    }
    pub(crate) fn push_block(&mut self, x: usize, y: usize, z: usize, ort: FaceOrientation, block: Block) {
        self.get_chunk_mut(x, z).push_block(x, y, z, ort, block)
    }
    pub(crate) fn remove_transparent_block_face(&mut self, x: usize, y: usize, z: usize, ort: FaceOrientation) {
        self.get_chunk_mut(x, z).remove_transparent_block_face(x, y, z, ort)
    }
    pub(crate) fn remove_opaque_block_face(&mut self, x: usize, y: usize, z: usize, ort: FaceOrientation) {
        self.get_chunk_mut(x, z).remove_opaque_block_face(x, y, z, ort)
    }
    pub(crate) fn update_block_textures(&mut self, x: usize, y: usize, z: usize, new_block: Block) {
        self.get_chunk_mut(x, z).update_block_textures(x, y, z, new_block)
    }
    pub(crate) fn change_block_textures(&mut self, x: usize, y: usize, z: usize, new_block: Block) {
        self.get_chunk_mut(x, z).change_block_textures(x, y, z, new_block)
    }
    pub fn get_chunk_mut(&mut self, x: usize, z: usize) -> &mut ChunkFaces {
        let u = self.size().block_pos_into_chunk_idx(x, z);
        &mut self.chunks[u]
    }
    pub fn get_chunk(&self, x: usize, z: usize) -> &ChunkFaces {
        &self.chunks[self.size().block_pos_into_chunk_idx(x, z)]
    }
}

impl Deref for WorldFaces{
    type Target = Vec<ChunkFaces>;

    fn deref(&self) -> &Self::Target {
        &self.chunks
    }
}


impl DerefMut for WorldFaces{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.chunks
    }
}