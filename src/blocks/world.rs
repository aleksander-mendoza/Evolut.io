use crate::blocks::world_blocks::WorldBlocks;
use crate::blocks::chunk_faces::{ChunkFaces};
use crate::blocks::world_size::{WorldSize, CHUNK_WIDTH, CHUNK_DEPTH, CHUNK_HEIGHT};
use crate::blocks::block::Block;
use crate::blocks::raycast::ray_cast;
use crate::blocks::world_faces::WorldFaces;
use crate::render::device::Device;
use crate::render::command_pool::{CommandBuffer, CommandPool};
use crate::render::submitter::Submitter;
use ash::prelude::VkResult;
use crate::render::pipeline::{PushConstant, Pipeline, BufferBinding};
use crate::blocks::Face;
use failure::err_msg;

pub struct World {
    blocks: WorldBlocks,
    faces: WorldFaces,
}
impl World {
    pub fn new(width: usize, depth: usize, cmd_pool:&CommandPool) -> Result<Submitter<Self>, failure::Error> {
        let size = WorldSize::new(width,depth);
        WorldFaces::new(size, cmd_pool.device()).map(|faces|Self{
            blocks: WorldBlocks::new(size),
            faces
        }).and_then(|w|Submitter::new(w, cmd_pool).map_err(err_msg))
    }
    pub fn blocks(&self) -> &WorldBlocks {
        &self.blocks
    }
    pub fn blocks_mut(&mut self) -> &mut WorldBlocks {
        &mut self.blocks
    }
    pub fn size(&self) -> &WorldSize {
        self.blocks.size()
    }

    pub fn update_set_block(&mut self, x: usize, y: usize, z: usize, block: Block) {
        self.update_block(x, y, z, move |b| {
            *b = block;
            true
        });
    }
    /**Returns true if previously there was no block at this position and the placement was carried out.
    If there was already a block, then placing a different one is impossible nad function returns false*/
    pub fn update_place_block(&mut self, x: usize, y: usize, z: usize, block: Block) -> bool {
        self.update_block(x, y, z, move |b| {
            if b.is_air() {
                *b = block;
                true
            } else {
                false
            }
        })
    }
    /**Returns true if previously there was block at this position and the removal was carried out.
    If there was no block, then no removal was necessary and function returns false*/
    pub fn update_remove_block(&mut self, x: usize, y: usize, z: usize) -> bool {
        self.update_block(x, y, z, move |b| {
            if !b.is_air() {
                *b = Block::air();
                true
            } else {
                false
            }
        })
    }
    /**Updates block according to custom policy. Function f should return true if a block was changed and face update is necessary.
    The result of this function is the same as the output of f.*/
    pub fn update_block<F: Fn(&mut Block) -> bool>(&mut self, x: usize, y: usize, z: usize, f: F) -> bool {
        let b = self.blocks_mut().get_block_mut(x, y, z);
        let was_showing_neighboring_faces = b.show_neighboring_faces();
        let was_showing_my_faces = b.show_my_faces();
        let was_transparent = b.is_transparent();
        if f(b) {
            let is_showing_neighboring_faces = b.show_neighboring_faces();
            let is_showing_my_faces = b.show_my_faces();
            let is_transparent = b.is_transparent();
            let b = b.clone();//just to make borrow-checker happy
            if was_showing_my_faces {
                if is_showing_my_faces {
                    if was_transparent == is_transparent {
                        self.faces.update_block_textures(x, y, z, b);
                    } else {
                        self.faces.change_block_textures(x, y, z, b);
                    }
                } else {
                    if was_transparent {
                        self.faces.remove_block_transparent(x, y, z);
                    } else {
                        self.faces.remove_block_opaque(x, y, z);
                    }
                }
            }

            self.size().clone().for_each_neighbour(x, y, z, |neighbour_x, neighbour_y, neighbour_z, my_face| {
                let &neighbour = self.blocks().get_block(neighbour_x, neighbour_y, neighbour_z);
                let neighbour_face = my_face.opposite();

                if was_showing_neighboring_faces && !is_showing_neighboring_faces && neighbour.show_my_faces() {
                    if neighbour.is_transparent() {
                        self.faces.remove_transparent_block_face(neighbour_x, neighbour_y, neighbour_z, neighbour_face)
                    } else {
                        self.faces.remove_opaque_block_face(neighbour_x, neighbour_y, neighbour_z, neighbour_face)
                    }
                }
                if !was_showing_neighboring_faces && is_showing_neighboring_faces && neighbour.show_my_faces() {
                    self.faces.push_block(neighbour_x, neighbour_y, neighbour_z, neighbour_face, neighbour);
                }
                if !was_showing_my_faces && is_showing_my_faces && neighbour.show_neighboring_faces() {
                    self.faces.push_block(x, y, z, my_face, b);
                }
            });
            true
        } else {
            false
        }
    }
    pub fn draw(&self, cmd: &mut CommandBuffer, pipeline:&Pipeline, instance_buffer_binding:BufferBinding<Face>,chunk_location_uniform: PushConstant<glm::Vec3>) {
        for (chunk_idx, chunk) in self.faces.iter().enumerate() {
            assert!(chunk_idx < self.faces.len());
            let (x, z) = self.size().chunk_idx_into_chunk_pos(chunk_idx);
            cmd.push_constant(pipeline,chunk_location_uniform,&glm::vec3((x * CHUNK_WIDTH) as f32, 0., (z * CHUNK_DEPTH) as f32))
                .vertex_input(instance_buffer_binding, chunk.opaque().gpu())
                .draw(6,chunk.len_opaque() as u32,0,0);
        }
        // TODO: cmd.barrier()?;
        for (chunk_idx, chunk) in self.faces.iter().enumerate() {
            assert!(chunk_idx < self.faces.len());
            let (x, z) = self.size().chunk_idx_into_chunk_pos(chunk_idx);
            cmd.push_constant(pipeline,chunk_location_uniform,&glm::vec3((x * CHUNK_WIDTH) as f32, 0., (z * CHUNK_DEPTH) as f32))
                .vertex_input(instance_buffer_binding, chunk.transparent().gpu())
                .draw(6,chunk.len_transparent() as u32,0,0);
        }
    }

    pub fn compute_faces(&mut self) {
        for x in 0..self.size().world_width() {
            for z in 0..self.size().world_depth() {
                for y in 0..CHUNK_HEIGHT {
                    let &block = self.blocks().get_block(x, y, z);
                    if block.show_my_faces() {
                        self.size().clone().for_each_neighbour(x, y, z, |neighbour_x, neighbour_y, neighbour_z, ort| {
                            let neighbour = self.blocks().get_block(neighbour_x, neighbour_y, neighbour_z);
                            if neighbour.show_neighboring_faces() {
                                self.faces.push_block(x, y, z, ort, block);
                            }
                        });
                    }
                }
            }
        }
    }

    pub fn ray_cast_place_block(&mut self, start: &[f32], distance_and_direction: &[f32], block: Block) {
        ray_cast(start, distance_and_direction, |block_x, block_y, block_z, adjacent_x, adjacent_y, adjacent_z| {
            if self.size().is_point_in_bounds(block_x, block_y, block_z) && !self.blocks().get_block(block_x as usize, block_y as usize, block_z as usize).is_air() {
                if block_x != adjacent_x || block_y != adjacent_y || block_z != adjacent_z {
                    let adjacent_y = adjacent_y as usize;
                    if adjacent_y < CHUNK_HEIGHT {//we don't need to test other coordinates because
                        // normally it should be impossible for a player to reach them
                        self.update_place_block(adjacent_x as usize, adjacent_y, adjacent_z as usize, block);
                    }
                }
                Some(())
            } else {
                None
            }
        });
    }

    pub fn ray_cast_remove_block(&mut self, start: &[f32], distance_and_direction: &[f32]) {
        ray_cast(start, distance_and_direction, |block_x, block_y, block_z, adjacent_x, adjacent_y, adjacent_z| {
            if self.size().is_point_in_bounds(block_x, block_y, block_z) &&
                self.update_remove_block(block_x as usize, block_y as usize, block_z as usize) {
                Some(())
            } else {
                None
            }
        });
    }
}

impl Submitter<World>{
    pub fn flush_all_chunks(&mut self) -> Result<(),failure::Error>{
        let (cmd, world) = self.inner_val();
        cmd.reset()?
            .reset()?
            .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        for chunk in world.faces.iter_mut() {
            chunk.flush_opaque(cmd.cmd());
            chunk.flush_transparent(cmd.cmd());
        }
        cmd.cmd().end()?;
        cmd.submit().map_err(err_msg)
    }
}