use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, IndirectDispatchSubBuffer, IndirectSubBuffer};
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool};


use ash::vk;


use failure::Error;


use crate::render::submitter::{Submitter, fill_submit};

use crate::render::buffer_type::{Cpu, Storage, GpuIndirect};

use crate::blocks::world_size::CHUNK_VOLUME_IN_CELLS;
use crate::render::subbuffer::SubBuffer;
use crate::pipelines::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::pipelines::particle_constants::ParticleConstants;
use crate::blocks::{WorldSize, Block, Face, WorldBlocks, WorldFaces};
use crate::render::sampler::Sampler;
use crate::pipelines::bone::Bone;
use crate::blocks::block_properties::{BLOCKS, BlockProp, BEDROCK, DIRT, GRASS, GLASS, PLANK, AIR};

pub struct Indirect {
    collision_detection: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    solve_constraints: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    update_bones: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    draw_bones: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_blocks: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_particles: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    super_indirect_buffer: SubBuffer<u8, GpuIndirect>
}

impl Indirect {
    fn new(super_indirect_buffer: SubBuffer<u8, GpuIndirect>, indirect_dispatch: &Submitter<IndirectDispatchSubBuffer>, indirect_draw: &Submitter<IndirectSubBuffer>) -> Self {
        let collision_detection = indirect_dispatch.gpu().element(0);
        let solve_constraints = indirect_dispatch.gpu().element(1);
        let update_bones = indirect_dispatch.gpu().element(2);
        let draw_bones = indirect_draw.gpu().element(0);
        let draw_blocks = indirect_draw.gpu().element(1);
        let draw_particles = indirect_draw.gpu().element(2);
        Self {
            super_indirect_buffer,
            collision_detection,
            solve_constraints,
            update_bones,
            draw_bones,
            draw_blocks,
            draw_particles,
        }
    }
    pub fn super_buffer(&self) -> &SubBuffer<u8, GpuIndirect> {
        &self.super_indirect_buffer
    }
    pub fn draw_bones(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_bones
    }
    pub fn draw_blocks(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_blocks
    }
    pub fn draw_particles(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_particles
    }
    pub fn collision_detection(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.collision_detection
    }
    pub fn solve_constraints(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.solve_constraints
    }
    pub fn update_bones(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.update_bones
    }
}

pub struct FoundationInitializer {
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    world: Submitter<StageSubBuffer<Block,Cpu,Storage>>,
    faces: Submitter<StageSubBuffer<Face,Cpu,Storage>>,
    face_count_per_chunk_buffer:SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer:SubBuffer<Face, Storage>,
    collision_grid: Submitter<SubBuffer<u32, Storage>>,
    constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    block_properties: Submitter<StageSubBuffer<BlockProp, Cpu, Storage>>,
    bones: Submitter<StageSubBuffer<Bone, Cpu, Storage>>,
    particle_constants: Submitter<StageSubBuffer<ParticleConstants, Cpu, Storage>>,
    indirect_dispatch: Submitter<IndirectDispatchSubBuffer>,
    indirect_draw: Submitter<IndirectSubBuffer>,
    indirect: Indirect,
    sampler: Sampler,
    world_size: WorldSize,
}

impl FoundationInitializer {
    pub fn face_count_per_chunk_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.opaque_and_transparent_face_buffer
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn particles(&self) -> &StageSubBuffer<Particle, Cpu, Storage> {
        &self.particles
    }
    pub fn constraints(&self) -> &StageSubBuffer<Constraint, Cpu, Storage> {
        &self.constraints
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
    pub fn bones(&self) -> &StageSubBuffer<Bone, Cpu, Storage> {
        &self.bones
    }
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.world_size
    }
    pub fn world(&self) -> &StageSubBuffer<Block, Cpu,Storage> {
        &self.world
    }
    pub fn faces(&self) -> &StageSubBuffer<Face, Cpu,Storage> {
        &self.faces
    }

    pub fn block_properties(&self) -> &StageSubBuffer<BlockProp, Cpu, Storage> {
        &self.block_properties
    }
    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let world_size = WorldSize::new(1,1);
        let particles = 512u64;
        let bones = 128u64;
        let faces = 2048u64*world_size.total_chunks() as u64;
        let max_constraints = 128u64;
        let grid_size = CHUNK_VOLUME_IN_CELLS as u64;
        let solid_particles = 32;
        let phantom_particles = 8;
        debug_assert!(solid_particles + phantom_particles <= particles);

        let mut world_blocks = WorldBlocks::new(world_size.clone());
        world_blocks.no_update_fill_level(0, 1, BEDROCK);
        world_blocks.no_update_fill_level(1, 1, DIRT);
        world_blocks.no_update_fill_level(2, 1, GRASS);
        world_blocks.no_update_fill_level(10, 1, GLASS);
        world_blocks.no_update_outline(5, 2, 5, 5, 5, 5, PLANK);
        // world_blocks.no_update_set_block(1,2,1,AIR);
        let mut world_faces = WorldFaces::with_capacity(world_size.clone(), faces as usize);
        world_faces.generate_faces(&world_blocks);
        let w2 = 0.4f32;
        let w = w2/2.;
        let h2 = 0.4f32;
        let h = h2/2.;
        let l = 0.6f32;
        let s = 0.2f32;
        let diag = (w2 * w2 + h2 * h2).sqrt();
        let diag_l = (w2 * w2 +l*l).sqrt();
        let diag_sl = (s*s+l*l).sqrt();
        let diag_wl = (w*w+l*l).sqrt();
        let mut particles_data: Vec<Particle> = std::iter::repeat_with(Particle::random).take((solid_particles+phantom_particles) as usize).collect();
        particles_data[1].new_position = glm::vec3(2., 7., 2.);
        particles_data[1].old_position = particles_data[1].new_position;
        particles_data[2].new_position = particles_data[1].new_position + glm::vec3(w2, 0., 0.);
        particles_data[2].old_position = particles_data[2].new_position;
        particles_data[3].new_position = particles_data[1].new_position + glm::vec3(w2, l, 0.);
        particles_data[3].old_position = particles_data[3].new_position;
        particles_data[4].new_position = particles_data[1].new_position + glm::vec3(0., l, 0.);
        particles_data[4].old_position = particles_data[4].new_position;
        particles_data[5].new_position = particles_data[1].new_position + glm::vec3(w2, h2 +l, 0.);
        particles_data[5].old_position = particles_data[5].new_position;
        particles_data[6].new_position = particles_data[1].new_position + glm::vec3(0., h2 +l, 0.);
        particles_data[6].old_position = particles_data[6].new_position;
        particles_data[7].new_position = particles_data[1].new_position + glm::vec3(w2 +s, l, 0.);
        particles_data[7].old_position = particles_data[7].new_position;
        particles_data[8].new_position = particles_data[1].new_position + glm::vec3(-s, l, 0.);
        particles_data[8].old_position = particles_data[8].new_position;
        particles_data[9].new_position = particles_data[1].new_position + glm::vec3(-s, 0., 0.);
        particles_data[9].old_position = particles_data[9].new_position;
        particles_data[10].new_position = particles_data[1].new_position + glm::vec3(w2 +s, 0., 0.);
        particles_data[10].old_position = particles_data[10].new_position;
        particles_data[11].new_position = particles_data[1].new_position + glm::vec3(0., -l, 0.);
        particles_data[11].old_position = particles_data[11].new_position;
        particles_data[12].new_position = particles_data[1].new_position + glm::vec3(w2, -l, 0.);
        particles_data[12].old_position = particles_data[12].new_position;
        particles_data[13].new_position = particles_data[1].new_position + glm::vec3(w, 0., 0.);
        particles_data[13].old_position = particles_data[13].new_position;
        particles_data[solid_particles as usize+0].new_position = particles_data[1].new_position + glm::vec3(0., 0., 0.);
        particles_data[solid_particles as usize+0].old_position = particles_data[solid_particles as usize+0].new_position;
        particles_data[solid_particles as usize+1].new_position = particles_data[1].new_position + glm::vec3(w2, 0., 0.);
        particles_data[solid_particles as usize+1].old_position = particles_data[solid_particles as usize +1].new_position;
        particles_data[solid_particles as usize+2].new_position = particles_data[1].new_position + glm::vec3(w, -l, 0.);
        particles_data[solid_particles as usize+2].old_position = particles_data[solid_particles as usize +2].new_position;
        particles_data[solid_particles as usize+3].new_position = particles_data[1].new_position + glm::vec3(w, -l, 0.);
        particles_data[solid_particles as usize+3].old_position = particles_data[solid_particles as usize +3].new_position;

        let predefined_constraints = vec![
            Constraint::distance(1, 2, w2),
            Constraint::distance(2, 3, l),
            Constraint::distance(3, 4, w2),
            Constraint::distance(4, 1, l),
            Constraint::distance(4, 2,diag_l),
            Constraint::distance(1, 3,diag_l),
            Constraint::distance(6, 5, w2),
            Constraint::distance(5, 3, h2),
            Constraint::distance(6, 4, w2),
            Constraint::distance(4, 5, diag),
            Constraint::distance(6, 3, diag),
            Constraint::distance(3, 7, s),
            Constraint::distance(7, 10, l),
            Constraint::distance(10, solid_particles as u32+1, s),
            Constraint::distance(solid_particles as u32+1, 3,l),
            Constraint::distance(solid_particles as u32+1, 7, diag_sl),
            Constraint::distance(3, 10, diag_sl),
            Constraint::distance(4, 8, s),
            Constraint::distance(8, 9, l),
            Constraint::distance(9, solid_particles as u32+0, s),
            Constraint::distance(solid_particles as u32+0, 4,l),
            Constraint::distance(solid_particles as u32+0, 8,diag_sl),
            Constraint::distance(9, 4,diag_sl),
            Constraint::distance(11, solid_particles as u32+2,w),
            Constraint::distance(solid_particles as u32+2,13,l),
            Constraint::distance(solid_particles as u32+2,1,diag_wl),
            Constraint::distance(11, 13,diag_wl),
            Constraint::distance(13,1,w),
            Constraint::distance(1,11,l),
            Constraint::distance(solid_particles as u32+3,12,w),
            Constraint::distance(12,2, l),
            Constraint::distance(2,13, w),
            Constraint::distance(13, solid_particles as u32+3,l),
            Constraint::distance(2, solid_particles as u32+3,diag_wl),
            Constraint::distance(12, 13,diag_wl),
        ];

        let block_properties_data:Vec<BlockProp> = BLOCKS.iter().map(|p|p.prop).collect();

        let bone_data = vec![
            Bone::new([1, 2, 3, 4], 2, 0.1),
            Bone::new([4, 3, 5, 6], 3, 0.2),
            Bone::new([solid_particles as u32+1, 10, 7, 3], 4, 0.1),
            Bone::new([9,solid_particles as u32+0, 4, 8], 5, 0.1),
            Bone::new([11,solid_particles as u32+2, 13, 1], 0, 0.1),
            Bone::new([solid_particles as u32+3, 12,2,13], 1, 0.1),
        ];

        let constants = ParticleConstants {
            predefined_constraints: predefined_constraints.len() as i32,
            collision_constraints: 0,
            solid_particles: solid_particles as i32,
            phantom_particles: phantom_particles as i32,
            chunks_x: world_size.width() as i32,
            chunks_z: world_size.depth() as i32,
            bones: bone_data.len() as i32,
            world_width: world_size.world_width()as i32,
            world_depth: world_size.world_depth() as i32,
            world_area: world_size.world_area()as i32,
            total_chunks: world_size.total_chunks()as i32,
            dummy1: 0
        };

        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * particles;
        let faces_in_bytes = std::mem::size_of::<Face>() as u64 * faces;
        let grid_in_bytes = std::mem::size_of::<u32>() as u64 * grid_size;
        let block_properties_in_bytes = std::mem::size_of_val(block_properties_data.as_slice()) as u64;
        let world_in_bytes = (std::mem::size_of::<Block>()*world_size.world_volume()) as u64;
        let constraints_in_bytes = std::mem::size_of::<Constraint>() as u64 * max_constraints;
        let bones_in_bytes = std::mem::size_of::<Bone>() as u64 * bones;
        let constants_in_bytes = std::mem::size_of_val(&constants) as u64;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            particles_in_bytes +
                                                                                faces_in_bytes +
                                                                                grid_in_bytes +
                                                                                block_properties_in_bytes +
                                                                                world_in_bytes +
                                                                                constraints_in_bytes +
                                                                                bones_in_bytes +
                                                                                constants_in_bytes)?;
        let offset = 0;
        let particle_buffer = super_buffer.sub(offset..offset + particles_in_bytes).reinterpret_into::<Particle>();
        let offset = offset + particles_in_bytes;
        assert_eq!(offset%16,0); // check correct GLSL alignment
        let face_buffer = super_buffer.sub(offset..offset + faces_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_in_bytes;
        assert_eq!(offset%16,0);
        let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        let offset = offset + grid_in_bytes;
        assert_eq!(offset%16,0);
        let block_properties_buffer = super_buffer.sub(offset..offset + block_properties_in_bytes).reinterpret_into::<BlockProp>();
        let offset = offset + block_properties_in_bytes;
        assert_eq!(offset%16,0);
        let world_buffer = super_buffer.sub(offset..offset + world_in_bytes).reinterpret_into::<Block>();
        let offset = offset + world_in_bytes;
        assert_eq!(offset%16,0);
        let constraint_buffer = super_buffer.sub(offset..offset + constraints_in_bytes).reinterpret_into::<Constraint>();
        let offset = offset + constraints_in_bytes;
        assert_eq!(offset%16,0);
        let bones_buffer = super_buffer.sub(offset..offset + bones_in_bytes).reinterpret_into::<Bone>();
        let offset = offset + bones_in_bytes;
        assert_eq!(offset%16,0);
        let constants_buffer = super_buffer.sub(offset..offset + constants_in_bytes).reinterpret_into::<ParticleConstants>();
        let offset = offset + constants_in_bytes;
        assert_eq!(offset%16,0);

        let particle_constants = StageBuffer::wrap(cmd_pool, &[constants], constants_buffer)?;

        let particles = StageBuffer::wrap(cmd_pool, &particles_data, particle_buffer)?;

        let face_count_per_chunk_buffer = face_buffer.sub(..std::mem::size_of::<Face>() as u64 *world_size.total_chunks() as u64*2);
        let opaque_and_transparent_face_buffer = face_buffer.sub(std::mem::size_of::<Face>() as u64  * world_size.total_chunks() as u64*2..);
        let faces = StageBuffer::wrap(cmd_pool,  world_faces.as_slice(), face_buffer)?;

        let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        fill_submit(&mut collision_grid, u32::MAX)?;

        let block_properties = StageBuffer::wrap(cmd_pool, &block_properties_data, block_properties_buffer)?;

        let bones = StageBuffer::wrap(cmd_pool, &bone_data, bones_buffer)?;

        let world = StageBuffer::wrap(cmd_pool,  world_blocks.as_slice(), world_buffer)?;

        let constraints = StageBuffer::wrap(cmd_pool, &predefined_constraints, constraint_buffer)?;

        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        fn dispatch_indirect(x: f32) -> vk::DispatchIndirectCommand {
            vk::DispatchIndirectCommand {
                x: (x / 32.).ceil() as u32,
                y: 1,
                z: 1,
            }
        }
        fn draw_indirect(vertex_count: u32, instance_count: u32) -> vk::DrawIndirectCommand {
            vk::DrawIndirectCommand {
                vertex_count,
                instance_count,
                first_vertex: 0,
                first_instance: 0,
            }
        }
        let indirect_dispatch_data = vec![
            dispatch_indirect(phantom_particles.max(solid_particles) as f32),// collision_detection.comp
            dispatch_indirect(0.), // solve_constraints.comp
            dispatch_indirect(bone_data.len() as f32) // update_bones.comp
        ];
        let indirect_draw_data = vec![
            draw_indirect(36, bone_data.len() as u32),// bones.vert
            draw_indirect(6, world_faces.len() as u32),// block.vert
            draw_indirect((solid_particles+phantom_particles) as u32, 1),// particles.vert
        ];
        let indirect_dispatch_in_bytes = std::mem::size_of_val(indirect_dispatch_data.as_slice()) as u64;
        let indirect_draw_in_bytes = std::mem::size_of_val(indirect_draw_data.as_slice()) as u64;
        let super_indirect_buffer: SubBuffer<u8, GpuIndirect> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                                         indirect_dispatch_in_bytes +
                                                                                             indirect_draw_in_bytes)?;
        let offset = 0;
        let indirect_dispatch_buffer = super_indirect_buffer.sub(offset..offset + indirect_dispatch_in_bytes).reinterpret_into::<vk::DispatchIndirectCommand>();
        let offset = offset + indirect_dispatch_in_bytes;
        let indirect_draw_buffer = super_indirect_buffer.sub(offset..offset + indirect_draw_in_bytes).reinterpret_into::<vk::DrawIndirectCommand>();
        let offset = offset + indirect_draw_in_bytes;

        let indirect_dispatch = StageBuffer::wrap(cmd_pool, &indirect_dispatch_data, indirect_dispatch_buffer)?;
        let indirect_draw = StageBuffer::wrap(cmd_pool, &indirect_draw_data, indirect_draw_buffer)?;

        let indirect = Indirect::new(super_indirect_buffer, &indirect_dispatch, &indirect_draw);

        Ok(Self {
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            faces,
            world,
            world_size,
            sampler,
            particles,
            constraints,
            collision_grid,
            particle_constants,
            indirect_dispatch,
            indirect_draw,
            indirect,
            bones,
        })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            indirect_dispatch,
            indirect_draw,
            world_size,
            faces,
            world,
            bones,
            particles,
            collision_grid,
            constraints,
            particle_constants,
            indirect,
            sampler,
        } = self;
        let particles = particles.take()?.take_gpu();
        let collision_grid = collision_grid.take()?;
        let block_properties = block_properties.take()?.take_gpu();
        let constraints = constraints.take()?.take_gpu();
        let bones = bones.take()?.take_gpu();
        let particle_constants = particle_constants.take()?.take_gpu();
        let _ = indirect_dispatch.take()?.take_gpu();
        let _ = indirect_draw.take()?.take_gpu();
        let world = world.take()?.take_gpu(); //wait for completion and then dispose of the staging buffer
        let faces = faces.take()?.take_gpu();
        Ok(Foundations {
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            block_properties,
            faces,
            world,
            world_size,
            bones,
            collision_grid,
            particle_constants,
            constraints,
            particles,
            indirect,
            sampler,
        })
    }
}

pub struct Foundations {
    world_size: WorldSize,
    block_properties: SubBuffer<BlockProp, Storage>,
    faces: SubBuffer<Face, Storage>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    world: SubBuffer<Block, Storage>,
    particles: SubBuffer<Particle, Storage>,
    constraints: SubBuffer<Constraint, Storage>,
    bones: SubBuffer<Bone, Storage>,
    particle_constants: SubBuffer<ParticleConstants, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
    indirect: Indirect,
    sampler: Sampler,
}

impl Foundations {
    pub fn face_count_per_chunk_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self)->&SubBuffer<Face, Storage>{
        &self.opaque_and_transparent_face_buffer
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.world_size
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particles
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn world(&self) -> &SubBuffer<Block, Storage> {
        &self.world
    }
    pub fn block_properties(&self) -> &SubBuffer<BlockProp, Storage> {
        &self.block_properties
    }
    pub fn faces(&self) -> &SubBuffer<Face, Storage> {
        &self.faces
    }
    pub fn constants(&self) -> &SubBuffer<ParticleConstants, Storage> {
        &self.particle_constants
    }
    pub fn constraints(&self) -> &SubBuffer<Constraint, Storage> {
        &self.constraints
    }

    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}