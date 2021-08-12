use crate::blocks::block::Block;
use crate::blocks::world_size::{WorldSize, CHUNK_WIDTH, CHUNK_DEPTH, CHUNK_HEIGHT};
use crate::blocks::{WorldBlocks, Face, FaceOrientation};


pub struct WorldFaces {
    faces: Vec<Face>,
    size: WorldSize,
}


impl WorldFaces {
    pub fn size(&self) -> &WorldSize {
        &self.size
    }
    pub fn len(&self) -> usize{
        self.faces.len() - self.size().total_chunks()*2
    }
    pub fn as_slice(&self) -> &[Face] {
        &self.faces
    }
    pub fn with_capacity(size: WorldSize, max_faces: usize) -> Self {
        let mut faces = Vec::with_capacity(max_faces);
        Self { faces , size }
    }
    fn for_each_face(size: &WorldSize, world: &WorldBlocks, chunk_x: usize, chunk_z: usize, mut f: impl FnMut(FaceOrientation, bool, &Block, glm::TVec3<usize>, glm::TVec3<u8>)) {
        for x in 0..CHUNK_WIDTH {
            for y in 0..CHUNK_HEIGHT {
                for z in 0..CHUNK_DEPTH {
                    let block_pos_relative_to_chunk = glm::vec3(x as u8, y as u8, z as u8);
                    let block_pos = glm::vec3(CHUNK_WIDTH * chunk_x + x, y, chunk_z * CHUNK_DEPTH + z);
                    let block = world.get_block(block_pos.x, block_pos.y, block_pos.z);
                    let world_bounds = [size.world_width(), size.height(), size.world_depth()];
                    for dim in 0..3 {
                        if block_pos[dim] + 1 < world_bounds[dim] {
                            let mut neighbour_pos = block_pos.clone();
                            neighbour_pos[dim] += 1;
                            let neighbour = world.get_block(neighbour_pos.x, neighbour_pos.y, neighbour_pos.z);
                            if block.opacity() != neighbour.opacity() { // If two neighboring blocks differ in opacities, then
                                // we must put a block face on the more opaque block, facing towards the less opaque block.
                                // If either block_prop.opacity==1 or neighbour_prop.opacity==1 then it's obvious that the face will be opaque.
                                // If both blocks are transparent (have opacity less than 1.0), then the face itself must be transparent.
                                // Chunks offsets for opaque faces are stored at faces[0..total_chunks]
                                // Chunks offsets for transparent faces are stored at faces[total_chunks..2*total_chunks]
                                let orientation = FaceOrientation::from_dim(dim, block.opacity() > neighbour.opacity());
                                let is_opaque = block.opacity() == 1.0 || neighbour.opacity() == 1.0;
                                let b = if block.opacity() > neighbour.opacity() { block}else{neighbour};
                                let pos = if block.opacity() > neighbour.opacity() {block_pos}else{neighbour_pos };
                                let mut pos_relative_to_chunk = block_pos_relative_to_chunk;
                                if block.opacity() < neighbour.opacity() {
                                    pos_relative_to_chunk[dim] += 1;
                                }
                                f(orientation, is_opaque,b, pos, pos_relative_to_chunk)
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn generate_faces(&mut self, world: &WorldBlocks) {
        let mut total_opaque_faces = 0;
        let mut total_transparent_faces = 0;
        let total_chunks = self.size().total_chunks();
        unsafe{
            self.faces.set_len(2*total_chunks);
            //Safety: Face implements Copy trait. We also don't really care about reading uninitialised junk anyway
        }
        for chunk_z in 0..self.size().depth() {
            for chunk_x in 0..self.size().width() {
                let current_chunk_idx = self.size().chunk_pos_into_chunk_idx(chunk_x, chunk_z);
                let mut opaque_faces = 0;
                let mut transparent_faces = 0;
                Self::for_each_face(self.size(), world, chunk_x, chunk_z, |orientation, is_opaque,block, _, _|
                    if is_opaque {
                        opaque_faces += 1;
                    } else {
                        transparent_faces += 1;
                    },
                );
                *self.faces[current_chunk_idx].as_mut_u32() = opaque_faces as u32;
                *self.faces[current_chunk_idx + total_chunks].as_mut_u32() = transparent_faces as u32;
                total_opaque_faces += opaque_faces;
                total_transparent_faces += transparent_faces;
            }
        }
        let mut opaque_face_offset = 2*total_chunks;
        let mut transparent_face_offset = opaque_face_offset+total_opaque_faces;
        unsafe{
            self.faces.set_len(transparent_face_offset+total_transparent_faces);
        }
        for chunk_z in 0..self.size().depth() {
            for chunk_x in 0..self.size().width() {
                let current_chunk_idx = self.size().chunk_pos_into_chunk_idx(chunk_x, chunk_z);
                let Self{ faces, size } = self;
                Self::for_each_face(size, world, chunk_x, chunk_z, |orientation, is_opaque, block, _, pos| {
                    let face = Face::from_coords_and_ort(chunk_x as u8, chunk_z as u8, pos[0], pos[1], pos[2], orientation, block.texture_id(orientation) as u16);
                    if is_opaque {
                        faces[opaque_face_offset] = face;
                        opaque_face_offset += 1;
                    } else {
                        faces[transparent_face_offset] = face;
                        transparent_face_offset += 1;
                    }
                });
            }
        }
    }
}


