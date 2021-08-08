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
use crate::blocks::WorldSize;
use crate::render::sampler::Sampler;
use crate::pipelines::bone::Bone;

pub struct Indirect{
    collision_detection:SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    solve_constraints:SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    update_bones:SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    draw_bones:SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
}
impl Indirect{
    fn new(indirect_dispatch:&Submitter<IndirectDispatchSubBuffer>, indirect_draw:&Submitter<IndirectSubBuffer>)->Self{
        let collision_detection = indirect_dispatch.gpu().element(0);
        let solve_constraints = indirect_dispatch.gpu().element(1);
        let update_bones = indirect_dispatch.gpu().element(2);
        let draw_bones = indirect_draw.gpu().element(0);
        Self{
            collision_detection,
            solve_constraints,
            update_bones,
            draw_bones
        }
    }
    pub fn draw_bones(&self)->&SubBuffer<vk::DrawIndirectCommand, GpuIndirect>{
        &self.draw_bones
    }
    pub fn collision_detection(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.collision_detection
    }
    pub fn solve_constraints(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.solve_constraints
    }
    pub fn update_bones(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.update_bones
    }
}
pub struct FoundationInitializer {
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    collision_grid: Submitter<SubBuffer<u32, Storage>>,
    constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    bones: Submitter<StageSubBuffer<Bone, Cpu, Storage>>,
    particle_constants: Submitter<StageSubBuffer<ParticleConstants, Cpu, Storage>>,
    indirect_dispatch: Submitter<IndirectDispatchSubBuffer>,
    indirect_draw: Submitter<IndirectSubBuffer>,
    indirect:Indirect,
    sampler: Sampler,
    world_size:WorldSize,
}

impl FoundationInitializer {
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

    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let world_size = WorldSize::new(2,2);
        let particles = 512u64;
        let bones = 128u64;
        let max_constraints = 128u64;
        let grid_size = CHUNK_VOLUME_IN_CELLS as u64;
        let solid_particles = 256;
        let phantom_particles = 256;
        debug_assert!(solid_particles + phantom_particles <= particles);

        let mut particles_data: Vec<Particle> = std::iter::repeat_with(Particle::random).take(particles as usize).collect();
        particles_data[1].new_position = glm::vec3(2., 7., 2.);
        particles_data[1].old_position = particles_data[1].new_position;
        particles_data[2].new_position = particles_data[1].new_position + glm::vec3(0.3, 0., 0.);
        particles_data[2].old_position = particles_data[2].new_position;
        particles_data[3].new_position = particles_data[1].new_position + glm::vec3(0.3, 0.3, 0.);
        particles_data[3].old_position = particles_data[3].new_position;
        particles_data[4].new_position = particles_data[1].new_position + glm::vec3(0., 0.3, 0.);
        particles_data[4].old_position = particles_data[4].new_position;

        let d = 0.4;
        let predefined_constraints = vec![
            Constraint::distance(1, 2, d),
            Constraint::distance(2, 3, d),
            Constraint::distance(3, 4, d),
            Constraint::distance(4, 1, d),
            Constraint::distance(4, 2, d * 2f32.sqrt()),
        ];

        let bone_data = vec![
            Bone::new([1,2,3,4], 3)
        ];

        let constants = ParticleConstants {
            predefined_constraints: predefined_constraints.len() as i32,
            collision_constraints: 0,
            solid_particles:solid_particles as i32,
            phantom_particles:phantom_particles as i32,
            chunks_x: world_size.width() as i32,
            chunks_z: world_size.depth() as i32,
            bones: bone_data.len() as i32,
            dummy: 0
        };

        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * particles;
        let grid_in_bytes = std::mem::size_of::<u32>() as u64 * grid_size;
        let constraints_in_bytes = std::mem::size_of::<Constraint>() as u64 * max_constraints;
        let bones_in_bytes = std::mem::size_of::<Bone>() as u64 * bones;
        let constants_in_bytes = std::mem::size_of_val(&constants) as u64;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            particles_in_bytes +
                                                                                grid_in_bytes +
                                                                                constraints_in_bytes +
                                                                                bones_in_bytes +
                                                                                constants_in_bytes)?;
        let offset = 0;
        let particle_buffer = super_buffer.sub(offset..offset+particles_in_bytes).reinterpret_into::<Particle>();
        let offset = offset+particles_in_bytes;
        let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        let offset = offset+grid_in_bytes;
        let constraint_buffer = super_buffer.sub(offset..offset + constraints_in_bytes).reinterpret_into::<Constraint>();
        let offset = offset+constraints_in_bytes;
        let bones_buffer = super_buffer.sub(offset..offset + bones_in_bytes).reinterpret_into::<Bone>();
        let offset = offset+bones_in_bytes;
        let constants_buffer = super_buffer.sub(offset..offset + constants_in_bytes).reinterpret_into::<ParticleConstants>();
        let _offset = offset+constants_in_bytes;

        let particle_constants = StageBuffer::wrap(cmd_pool, &[constants], constants_buffer)?;

        let particles = StageBuffer::wrap(cmd_pool, &particles_data, particle_buffer)?;

        let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        fill_submit(&mut collision_grid,u32::MAX)?;

        let bones = StageBuffer::wrap(cmd_pool, &bone_data, bones_buffer)?;

        let constraints = StageBuffer::wrap(cmd_pool, &predefined_constraints, constraint_buffer)?;

        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        fn dispatch_indirect(x:f32)->vk::DispatchIndirectCommand{
            vk::DispatchIndirectCommand{
                x: (x/32.).ceil() as u32,
                y: 1,
                z: 1
            }
        }
        fn draw_indirect(vertex_count:u32, instance_count:u32)->vk::DrawIndirectCommand{
            vk::DrawIndirectCommand{
                vertex_count,
                instance_count,
                first_vertex: 0,
                first_instance: 0
            }
        }
        let indirect_dispatch_data = vec![
            dispatch_indirect(phantom_particles.max(solid_particles) as f32),// collision_detection.comp
            dispatch_indirect(0.), // solve_constraints.comp
            dispatch_indirect(bone_data.len() as f32) // bones.comp
        ];
        let indirect_draw_data = vec![
            draw_indirect(36, bone_data.len() as u32),// bones.vert
        ];
        let indirect_dispatch_in_bytes = std::mem::size_of_val(indirect_dispatch_data.as_slice()) as u64;
        let indirect_draw_in_bytes = std::mem::size_of_val(indirect_draw_data.as_slice()) as u64;
        let super_indirect_buffer: SubBuffer<u8, GpuIndirect> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                                     indirect_dispatch_in_bytes +
                                                                                         indirect_draw_in_bytes)?;
        let offset = 0;
        let indirect_dispatch_buffer = super_indirect_buffer.sub(offset..offset+indirect_dispatch_in_bytes).reinterpret_into::<vk::DispatchIndirectCommand>();
        let offset = offset+indirect_dispatch_in_bytes;
        let indirect_draw_buffer = super_indirect_buffer.sub(offset..offset + indirect_draw_in_bytes).reinterpret_into::<vk::DrawIndirectCommand>();
        let offset = offset+indirect_draw_in_bytes;

        let indirect_dispatch = StageBuffer::wrap(cmd_pool, &indirect_dispatch_data, indirect_dispatch_buffer)?;
        let indirect_draw = StageBuffer::wrap(cmd_pool, &indirect_draw_data, indirect_draw_buffer)?;

        let indirect = Indirect::new(&indirect_dispatch, &indirect_draw);

        Ok(Self { world_size, sampler, particles, constraints, collision_grid, particle_constants,
            indirect_dispatch, indirect_draw, indirect, bones })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
            indirect_dispatch,
            indirect_draw,
            world_size,
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
        let constraints = constraints.take()?.take_gpu();
        let bones = bones.take()?.take_gpu();
        let particle_constants = particle_constants.take()?.take_gpu();
        let indirect_dispatch = indirect_dispatch.take()?.take_gpu();
        let indirect_draw = indirect_draw.take()?.take_gpu();
        Ok(Foundations {
            indirect_draw,
            indirect_dispatch,
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
    particles: SubBuffer<Particle, Storage>,
    constraints: SubBuffer<Constraint, Storage>,
    bones: SubBuffer<Bone, Storage>,
    particle_constants: SubBuffer<ParticleConstants, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
    indirect_dispatch:  SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    indirect_draw:  SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    indirect:Indirect,
    sampler: Sampler,
}

impl Foundations {
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particles
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn constants(&self) -> &SubBuffer<ParticleConstants, Storage> {
        &self.particle_constants
    }
    pub fn indirect_dispatch(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.indirect_dispatch
    }
    pub fn indirect_draw(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.indirect_draw
    }
    pub fn constraints(&self) -> &SubBuffer<Constraint, Storage> {
        &self.constraints
    }

    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}