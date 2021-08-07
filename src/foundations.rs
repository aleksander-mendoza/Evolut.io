use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, StageOwnedBuffer, IndirectDispatchSubBuffer};
use crate::particle::Particle;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Fragment, Vertex};
use ash::vk::ShaderStageFlags;
use crate::render::pipeline::{PipelineBuilder, BufferBinding, Pipeline};
use ash::vk;
use crate::display::{Resources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::render::submitter::{Submitter, fill_submit};
use crate::player::Player;
use crate::render::buffer_type::{Cpu, Storage, GpuIndirect};
use crate::render::owned_buffer::OwnedBuffer;
use crate::blocks::world_size::CHUNK_VOLUME_IN_CELLS;
use crate::render::subbuffer::SubBuffer;
use crate::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::particle_constants::ParticleConstants;
use crate::blocks::WorldSize;
use crate::render::sampler::Sampler;
use crate::bone::Bone;

pub struct FoundationInitializer {
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    collision_grid: Submitter<SubBuffer<u32, Storage>>,
    constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    bones: Submitter<StageSubBuffer<Bone, Cpu, Storage>>,
    particle_constants: Submitter<StageSubBuffer<ParticleConstants, Cpu, Storage>>,
    indirect: Submitter<IndirectDispatchSubBuffer>,
    sampler: Sampler,
    world_size:WorldSize,
}

impl FoundationInitializer {
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
            Bone::new(glm::vec4(1,2,3,4))
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
        let offset = offset+constants_in_bytes;

        let particle_constants = StageBuffer::wrap(cmd_pool, &[constants], constants_buffer)?;

        let particles = StageBuffer::wrap(cmd_pool, &particles_data, particle_buffer)?;

        let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        fill_submit(&mut collision_grid,u32::MAX)?;

        let indirect = StageBuffer::new_indirect_dispatch_buffer(cmd_pool, &[
            vk::DispatchIndirectCommand { // solve_constraints.comp
                x: 0, // number of predefined_constraints + collision_constraints
                y: 1,
                z: 1,
            }, vk::DispatchIndirectCommand { // collision_detection.comp
                x: phantom_particles.max(solid_particles) as u32,
                y: 1,
                z: 1,
            }, vk::DispatchIndirectCommand {
                x: 0,
                y: 1,
                z: 1,
            }
        ])?;

        let bones = StageBuffer::wrap(cmd_pool, &bone_data, bones_buffer)?;

        let constraints = StageBuffer::wrap(cmd_pool, &predefined_constraints, constraint_buffer)?;

        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        Ok(Self { world_size, sampler, particles, constraints, collision_grid, particle_constants, indirect, bones })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
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
        let indirect = indirect.take()?.take_gpu();
        Ok(Foundations {
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
    indirect:  SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    sampler: Sampler,
}

impl Foundations {
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particles
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn constants(&self) -> &SubBuffer<ParticleConstants, Storage> {
        &self.particle_constants
    }
    pub fn indirect(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.indirect
    }
    pub fn constraints(&self) -> &SubBuffer<Constraint, Storage> {
        &self.constraints
    }

    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}