use crate::blocks::block_properties::{BLOCKS, STONE};
use std::fmt::{Display, Formatter};
use crate::blocks::block::Block;
use crate::blocks::face_orientation::FaceOrientation;
use crate::blocks::chunk_faces::ChunkFaces;
use crate::blocks::world_size::WorldSize;


pub struct WorldBlocks{
    blocks: Vec<Block>,
    size:WorldSize
}


impl WorldBlocks {
    pub fn size(&self)->&WorldSize{
        &self.size
    }
    pub fn as_slice(&self)->&[Block]{
        &self.blocks
    }
    pub fn get_block(&self, x: usize, y: usize, z: usize) -> &Block {
        &self.blocks[self.size().block_pos_into_world_idx(x,y,z)]
    }
    pub fn get_block_mut(&mut self, x: usize, y: usize, z: usize) -> &mut Block {
        let idx = self.size().block_pos_into_world_idx(x,y,z);
        &mut self.blocks[idx]
    }

    pub fn new(size: WorldSize) -> Self {
        Self { size, blocks:vec![Block::air();size.world_volume()] }
    }

    pub fn no_update_remove_block(&mut self, x: usize, y: usize, z: usize) {
        self.no_update_set_block(x, y, z, Block::air())
    }
    pub fn no_update_set_block(&mut self, x: usize, y: usize, z: usize, block: Block) {
        *self.get_block_mut(x, y, z) = block
    }
    pub fn no_update_fill(&mut self, from_x: usize, from_y: usize, from_z: usize, width: usize, height: usize, depth: usize, block: Block) {
        for x in from_x..(from_x + width) {
            for y in from_y..(from_y + height) {
                for z in from_z..(from_z + depth) {
                    self.no_update_set_block(x, y, z, block);
                }
            }
        }
    }
    pub fn no_update_outline(&mut self, from_x: usize, from_y: usize, from_z: usize, width: usize, height: usize, depth: usize, block: Block) {
        self.no_update_fill(from_x, from_y, from_z,width, 1, depth, block);
        if height > 1 {
            self.no_update_fill(from_x, from_y + height - 1, from_z , width, 1, depth, block);
            if height > 2 {
                for y in from_y+1..(from_y + height-1) {
                    if width>0 {
                        for z in from_z..(from_z + depth) {
                            self.no_update_set_block(from_x, y, z, block);
                            self.no_update_set_block(from_x + width - 1, y, z, block);
                        }
                    }
                    if depth>2 {
                        for x in from_x..(from_x + width) {
                            self.no_update_set_block(x, y, from_z, block);
                            self.no_update_set_block(x, y, from_z+depth-1, block);
                        }
                    }
                }
            }
        }

    }
    pub fn no_update_fill_level(&mut self, from_y: usize, height: usize, block: Block) {
        self.no_update_fill(0, from_y, 0, self.size().world_width(), height, self.size().world_depth(), block)
    }
    pub fn no_update_replace(&mut self, from_x: usize, from_y: usize, from_z: usize, width: usize, height: usize, depth: usize, old_block: Block, new_block: Block) {
        for x in from_x..(from_x + width) {
            for y in from_y..(from_y + height) {
                for z in from_z..(from_z + depth) {
                    let b = self.get_block_mut(x, y, z);
                    if b == &old_block {
                        *b = new_block
                    }
                }
            }
        }
    }
    pub fn no_update_heightmap(&mut self, filler_block: Block, height_at: impl Fn(usize, usize) -> usize) {
        for x in 0..self.size().world_width() {
            for z in 0..self.size().world_depth() {
                for y in 0..height_at(x, z) {
                    self.no_update_set_block(x, y, z, filler_block)
                }
            }
        }
    }
    pub fn contains_solid_block(&self, from_x: usize, from_y: usize, from_z: usize, width: usize, height: usize, depth: usize) ->bool{
        self.contains_solid_block_in(from_x,from_y,from_z,from_x+width,from_y+height,from_z+depth)
    }
    pub fn contains_solid_block_in(&self, from_x: usize, from_y: usize, from_z: usize, to_x: usize, to_y: usize, to_z: usize) ->bool{
        for x in from_x..to_x {
            for y in from_y..to_y {
                for z in from_z..to_z {
                    if self.get_block(x,y,z).is_solid(){
                        return true;
                    }
                }
            }
        }
        false
    }
    pub fn hitbox_collision(&self,hitbox_from:&glm::Vec3,hitbox_to:&glm::Vec3)->[bool;6]{
        let (start_x,start_y,start_z) = (hitbox_from[0] as usize,hitbox_from[1] as usize,hitbox_from[2] as usize);
        let (end_x,end_y,end_z) = (hitbox_to[0].ceil() as usize, hitbox_to[1].ceil() as usize, hitbox_to[2].ceil() as usize);
        let cant_go_minus_x = if start_x>0{self.contains_solid_block_in(start_x-1,start_y,start_z,start_x,end_y,end_z)}else{true};
        let cant_go_plus_x = if end_x<self.size().world_width(){self.contains_solid_block_in(end_x,start_y,start_z,end_x+1,end_y,end_z)}else{true};
        let cant_go_minus_y = if start_y>0{self.contains_solid_block_in(start_x,start_y-1,start_z,end_x,start_y,end_z)}else{true};
        let cant_go_plus_y = if end_y<self.size().height(){self.contains_solid_block_in(start_x,end_y,start_z,end_x,end_y+1,end_z)}else{true};
        let cant_go_minus_z = if start_z>0{self.contains_solid_block_in(start_x,start_y,start_z-1,end_x,end_y,start_z)}else{true};
        let cant_go_plus_z = if end_z<self.size().world_depth(){self.contains_solid_block_in(start_x,start_y,end_z,end_x,end_y,end_z+1)}else{true};
        [cant_go_minus_x,cant_go_plus_x,cant_go_minus_y,cant_go_plus_y,cant_go_minus_z,cant_go_plus_z]
    }

    pub fn zero_out_velocity_vector_on_hitbox_collision(&self,velocity:&mut glm::Vec3,hitbox_from:&glm::Vec3,hitbox_to:&glm::Vec3){
        let cant_go_in_direction = self.hitbox_collision(hitbox_from,hitbox_to);
        for axis in 0..3{
            if cant_go_in_direction[axis*2] && hitbox_from[axis]+velocity[axis] <= hitbox_from[axis].floor(){
                velocity[axis] = 0.
            }else if cant_go_in_direction[axis*2+1] && hitbox_to[axis].ceil() <= hitbox_to[axis]+velocity[axis]  {
                velocity[axis] = 0.
            }
        }
    }




}


