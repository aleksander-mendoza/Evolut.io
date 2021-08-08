use crate::blocks::face::Face;
use crate::blocks::block::Block;
use crate::blocks::face_orientation::FaceOrientation;
use crate::blocks::world_size::{WorldSize};
use crate::render::device::Device;


use crate::render::command_pool::CommandBuffer;
use failure::err_msg;
use crate::render::stage_buffer::{VertexBuffer, VertexOwnedBuffer};

pub struct ChunkFaces {
    opaque_faces: VertexOwnedBuffer<Face>,
    transparent_faces: VertexOwnedBuffer<Face>,
}

impl ChunkFaces {
    pub fn opaque(&self) -> &VertexOwnedBuffer<Face> {
        &self.opaque_faces
    }
    pub fn transparent(&self) -> &VertexOwnedBuffer<Face> {
        &self.transparent_faces
    }
    pub fn flush_opaque(&mut self, cmd:&mut CommandBuffer) {
        cmd.copy_from_staged_if_has_changes(&mut self.opaque_faces);
    }
    pub fn flush_transparent(&mut self, cmd: &mut CommandBuffer) {
        cmd.copy_from_staged_if_has_changes(&mut self.transparent_faces);
    }
    pub fn opaque_as_slice(&self) -> &[Face] {
        self.opaque_faces.as_slice()
    }
    pub fn transparent_as_slice(&self) -> &[Face] {
        self.transparent_faces.as_slice()
    }
    pub fn len_opaque(&self) -> usize {
        self.opaque_faces.len() as usize
    }
    pub fn len_transparent(&self) -> usize {
        self.transparent_faces.len() as usize
    }
    pub fn new(device:&Device) -> Result<Self,failure::Error> {
        Ok(Self {
            opaque_faces: VertexBuffer::with_capacity(device,16)?,
            transparent_faces: VertexBuffer::with_capacity(device,16)?,
        })
    }
    pub fn push_block(&mut self, x: usize, y: usize, z: usize, ort: FaceOrientation, block: Block) -> Result<bool, failure::Error> {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.push(x, y, z, ort, block)
    }
    fn push(&mut self, x: u8, y: u8, z: u8, ort: FaceOrientation, block: Block) -> Result<bool, failure::Error> {
        let face = Face::from_coords_and_ort(x, y, z, ort, block.texture_id(ort));
        debug_assert!(self.find_opaque_by_coords_and_ort(face.coords_and_ort()).is_none());
        debug_assert!(self.find_transparent_by_coords_and_ort(face.coords_and_ort()).is_none());
        if block.is_transparent() {
            self.transparent_faces.push(face).map_err(err_msg)
        } else {
            self.opaque_faces.push(face).map_err(err_msg)
        }
    }
    pub fn find_transparent_by_coords_and_ort(&self, coords: u32) -> Option<&Face> {
        self.transparent_faces.iter().find(|f| f.coords_and_ort() == coords)
    }
    pub fn find_opaque_by_coords_and_ort(&self, coords: u32) -> Option<&Face> {
        self.opaque_faces.iter().find(|f| f.coords_and_ort() == coords)
    }
    pub fn position_transparent_by_coords_and_ort(&self, coords: u32) -> Option<usize> {
        self.transparent_faces.iter().position(|f| f.coords_and_ort() == coords)
    }
    pub fn position_opaque_by_coords_and_ort(&self, coords: u32) -> Option<usize> {
        self.opaque_faces.iter().position(|f| f.coords_and_ort() == coords)
    }
    pub fn find_transparent(&self, x: u8, y: u8, z: u8) -> Option<&Face> {
        self.transparent_faces.iter().find(|f| f.matches_coords(x, y, z))
    }
    pub fn find_opaque(&self, x: u8, y: u8, z: u8) -> Option<&Face> {
        self.opaque_faces.iter().find(|f| f.matches_coords(x, y, z))
    }
    pub(crate) fn remove_block_transparent(&mut self, x: usize, y: usize, z: usize) {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.remove_transparent(x, y, z);
    }

    fn remove_transparent(&mut self, x: u8, y: u8, z: u8) {
        let mut i = 0;
        debug_assert!(self.find_opaque(x, y, z).is_none());
        debug_assert!(self.find_transparent(x, y, z).is_some());
        while i < self.len_transparent() {
            if self.transparent_faces[i].matches_coords(x, y, z) {
                self.remove_transparent_at(i);
            } else {
                i += 1;
            }
        }
    }
    pub(crate) fn remove_block_opaque(&mut self, x: usize, y: usize, z: usize) {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.remove_opaque(x, y, z)
    }
    fn remove_opaque(&mut self, x: u8, y: u8, z: u8) {
        debug_assert!(self.find_opaque(x, y, z).is_some());
        debug_assert!(self.find_transparent(x, y, z).is_none());
        let mut i = 0;
        while i < self.len_opaque() {
            if self.opaque_faces[i].matches_coords(x, y, z) {
                self.remove_opaque_at(i);
            } else {
                i += 1;
            }
        }
    }
    pub(crate) fn update_block_textures(&mut self, x: usize, y: usize, z: usize, new_block: Block) {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.update_textures(x, y, z, new_block)
    }
    /**The transparency of old textures must be the same as that of new ones. If transparency can change, use change_textures instead*/
    fn update_textures(&mut self, x: u8, y: u8, z: u8, new_block: Block) {
        assert!(!new_block.is_air());
        let faces = if new_block.is_transparent() {
            assert!(self.find_opaque(x, y, z).is_none(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            assert!(self.find_transparent(x, y, z).is_some(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            &mut self.transparent_faces
        } else {
            assert!(self.find_opaque(x, y, z).is_some(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            assert!(self.find_transparent(x, y, z).is_none(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            &mut self.opaque_faces
        };
        faces.mark_with_unflushed_changes();
        for face in faces.iter_mut() {
            if face.matches_coords(x, y, z) {
                face.update_texture(new_block);
            }
        }
    }
    fn borrow_transparent_and_opaque_mut(&mut self) -> (&mut VertexOwnedBuffer<Face>, &mut VertexOwnedBuffer<Face>) {
        let Self { transparent_faces, opaque_faces, .. } = self;
        (transparent_faces, opaque_faces)
    }
    pub fn change_block_textures(&mut self, x: usize, y: usize, z: usize, new_block: Block) -> Result<(), failure::Error> {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.change_textures(x, y, z, new_block)
    }
    /**Changes textures on existing faces and assumes that the transparency is going to be switched. If transparency did not change, use update_textures instead*/
    fn change_textures(&mut self, x: u8, y: u8, z: u8, new_block: Block) -> Result<(),failure::Error> {
        debug_assert!(!new_block.is_air());
        let (from, to) = if new_block.is_transparent() {
            debug_assert!(self.find_opaque(x, y, z).is_some(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            debug_assert!(self.find_transparent(x, y, z).is_none(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            let (trans, opaq) = self.borrow_transparent_and_opaque_mut();
            (opaq, trans)
        } else {
            debug_assert!(self.find_opaque(x, y, z).is_none(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            debug_assert!(self.find_transparent(x, y, z).is_some(), "Failed to update texture at {},{},{} to new block id {}", x, y, z, new_block);
            self.borrow_transparent_and_opaque_mut()
        };

        let mut i = 0;
        while i < from.len() as usize {
            if from[i].matches_coords(x, y, z) {
                to.push(from.swap_remove(i))?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }
    pub(crate) fn remove_opaque_block_face(&mut self, x: usize, y: usize, z: usize, ort: FaceOrientation) {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.remove_opaque_face(x, y, z, ort)
    }
    fn remove_opaque_face(&mut self, x: u8, y: u8, z: u8, ort: FaceOrientation) {
        let face = Face::encode_coords_and_ort(x, y, z, ort);
        self.remove_opaque_at(self.position_opaque_by_coords_and_ort(face).unwrap())
    }
    pub(crate) fn remove_transparent_block_face(&mut self, x: usize, y: usize, z: usize, ort: FaceOrientation) {
        let (x, y, z) = WorldSize::absolute_block_to_chunk_block_position(x,y,z);
        self.remove_transparent_face(x, y, z, ort)
    }
    fn remove_transparent_face(&mut self, x: u8, y: u8, z: u8, ort: FaceOrientation) {
        let face = Face::encode_coords_and_ort(x, y, z, ort);
        self.remove_transparent_at(self.position_transparent_by_coords_and_ort(face).unwrap())
    }
    fn update_texture(&mut self, idx: usize, new_block: Block) {
        debug_assert!(!new_block.is_air());
        let faces = if new_block.is_transparent() {
            &mut self.transparent_faces
        } else {
            &mut self.opaque_faces
        };
        faces.mark_with_unflushed_changes();
        faces[idx].update_texture(new_block)
    }
    fn remove_transparent_at(&mut self, idx: usize) {
        self.transparent_faces.swap_remove(idx);
    }
    fn remove_opaque_at(&mut self, idx: usize) {
        self.opaque_faces.swap_remove(idx);
    }
}



