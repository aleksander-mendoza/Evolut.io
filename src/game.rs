use crate::render::stage_buffer::{StageBuffer, IndirectDispatchBuffer};
use crate::particle::Particle;
use crate::render::owned_buffer::{OwnedBuffer};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Fragment, Vertex, Compute};
use ash::vk::ShaderStageFlags;
use crate::render::pipeline::{PipelineBuilder, BufferBinding, Pipeline};
use ash::vk;
use crate::display::{Resources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::render::submitter::Submitter;
use crate::joint::{Joint, JointResources};
use crate::particles::{ParticleResources, Particles};
use crate::block_world::{BlockWorldResources, BlockWorld};
use crate::render::compute::{ComputePipelineBuilder, ComputePipeline, StorageBufferBinding, UniformBufferBinding};
use crate::blocks::world_size::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, CHUNK_WIDTH_IN_CELLS, CHUNK_DEPTH_IN_CELLS, CHUNK_HEIGHT_IN_CELLS};
use crate::render::vector::Vector;
use crate::render::host_buffer::HostBuffer;
use crate::player::Player;
use crate::render::uniform_types::Vec3;
use crate::render::buffer_type::{Storage, Cpu, Uniform, GpuIndirect};
use crate::render::buffer::make_shader_buffer_barrier;

pub struct GameResources {
    res: JointResources<BlockWorldResources, ParticleResources>,
    collision_detection: ShaderModule<Compute>,
    solve_constraints: ShaderModule<Compute>,
    collision_grid: Submitter<OwnedBuffer<u32, Storage>>,
    indirect_buffer: Submitter<IndirectDispatchBuffer>,
    particle_constants: Submitter<StageBuffer<ParticleConstants, Cpu, Storage>>,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
    constraints_buffer: Submitter<StageBuffer<Constraint, Cpu, Storage>>,
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct ParticleConstants {
    predefined_constraints: u32,
    collision_constraints: u32,
    solid_particles: u32,
    phantom_particles: u32,
    chunks_x: u32,
    chunks_z: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Constraint {
    particle1: u32,
    particle2: u32,
    constant_param: f32,
}


#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
pub struct ThrowUniform {
    position: Vec3,
    velocity: Vec3,
}

impl Resources for GameResources {
    type Render = Game;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let res = JointResources::<BlockWorldResources, ParticleResources>::new(cmd_pool)?;
        let collision_detection = ShaderModule::new(include_glsl!("assets/shaders/collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let solve_constraints = ShaderModule::new(include_glsl!("assets/shaders/solve_constraints.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let mut collision_grid = Submitter::new(OwnedBuffer::with_capacity(cmd_pool.device(), CHUNK_WIDTH_IN_CELLS * CHUNK_HEIGHT_IN_CELLS * CHUNK_DEPTH_IN_CELLS)?, cmd_pool)?;
        collision_grid.fill_submit(u32::MAX)?;
        let world_size = res.a().world().size();
        let d = 0.4;
        let constraints_buffer = StageBuffer::new_with_capacity(cmd_pool, &[Constraint {
            particle1: 1,
            particle2: 2,
            constant_param: d,
        }, Constraint {
            particle1: 2,
            particle2: 3,
            constant_param: d,
        }, Constraint {
            particle1: 3,
            particle2: 4,
            constant_param: d,
        }, Constraint {
            particle1: 4,
            particle2: 1,
            constant_param: d,
        }, Constraint {
            particle1: 4,
            particle2: 2,
            constant_param: d * 2f32.sqrt(),
        }], 128)?;
        let solid_particles = 256;
        let phantom_particles = 256;
        let particle_constants = StageBuffer::new(cmd_pool, &[ParticleConstants {
            predefined_constraints: constraints_buffer.len() as u32,
            collision_constraints: 0,
            solid_particles,
            phantom_particles,
            chunks_x: world_size.width() as u32,
            chunks_z: world_size.depth() as u32,
        }])?;
        assert!((solid_particles+phantom_particles) as usize <= res.b().particles().capacity());
        let particle_uniform = HostBuffer::new(cmd_pool.device(), &[ThrowUniform {
            position: Vec3(glm::vec3(0., 0., 0.)),
            velocity: Vec3(glm::vec3(0., 0., 0.)),
        }])?;
        let indirect_buffer = StageBuffer::new_indirect_dispatch_buffer(cmd_pool, &[
            vk::DispatchIndirectCommand { // solve_constraints.comp
                x: 0, // number of predefined_constraints + collision_constraints
                y: 1,
                z: 1,
            }, vk::DispatchIndirectCommand { // collision_detection.comp
                x: phantom_particles.max(solid_particles),
                y: 1,
                z: 1,
            }, vk::DispatchIndirectCommand {
                x: 0,
                y: 1,
                z: 1,
            }
        ])?;
        Ok(Self { res, indirect_buffer, collision_detection, solve_constraints, collision_grid, particle_constants, particle_uniform, constraints_buffer })
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        self.res.create_descriptors(descriptors)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self {
            res,
            collision_detection,
            solve_constraints,
            collision_grid,
            mut particle_constants,
            particle_uniform,
            constraints_buffer,
            indirect_buffer,
        } = self;
        let global = res.make_renderable(cmd_pool, render_pass, descriptors)?;
        let mut compute_pipeline = ComputePipelineBuilder::new();
        let uniform_binding = compute_pipeline.uniform_buffer(particle_uniform.buffer());
        let constant_binding = compute_pipeline.storage_buffer(particle_constants.gpu());
        let particles_binding = compute_pipeline.storage_buffer(global.b().particles().gpu());
        let collision_grid_binding = compute_pipeline.storage_buffer(&collision_grid);
        let constraint_binding = compute_pipeline.storage_buffer(constraints_buffer.gpu());
        let indirect_binding = compute_pipeline.storage_buffer(indirect_buffer.gpu());
        compute_pipeline.shader("main", collision_detection);
        let collision_detection = compute_pipeline.build(cmd_pool.device())?;
        compute_pipeline.shader("main", solve_constraints);
        let solve_constraints = compute_pipeline.build(cmd_pool.device())?;
        let collision_grid = collision_grid.take()?;
        let constraints_buffer = constraints_buffer.take()?.take_gpu();
        let indirect_buffer = indirect_buffer.take()?.take_gpu();
        particle_constants.reset()?;
        Ok(Game {
            global,
            indirect_buffer,
            collision_detection,
            solve_constraints,
            particles_binding,
            collision_grid_binding,
            collision_grid,
            uniform_binding,
            constant_binding,
            particle_constants,
            particle_uniform,
            constraints_buffer,
            constraint_binding,
        })
    }
}


pub struct Game {
    //The order matters because of drop!
    uniform_binding: UniformBufferBinding<ThrowUniform>,
    constant_binding: StorageBufferBinding<ParticleConstants>,
    particles_binding: StorageBufferBinding<Particle>,
    constraint_binding: StorageBufferBinding<Constraint>,
    collision_grid_binding: StorageBufferBinding<u32>,
    collision_detection: ComputePipeline,
    solve_constraints: ComputePipeline,
    global: Joint<BlockWorld, Particles>,
    collision_grid: OwnedBuffer<u32, Storage>,
    particle_constants: Submitter<StageBuffer<ParticleConstants, Cpu, Storage>>,
    constraints_buffer: OwnedBuffer<Constraint, Storage>,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
    indirect_buffer: OwnedBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
}

impl Game {
    pub fn block_world(&self) -> &BlockWorld {
        self.global.a()
    }
    pub fn particles(&self) -> &Particles {
        self.global.b()
    }
    pub fn block_world_mut(&mut self) -> &mut BlockWorld {
        self.global.a_mut()
    }
    pub fn particles_mut(&mut self) -> &mut Particles {
        self.global.b_mut()
    }
}

impl Renderable for Game {
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors: &Descriptors, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.global.record_cmd_buffer(cmd, image_idx, descriptors, render_pass)?;
        Ok(())
    }

    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer) -> Result<(), Error> {
        cmd.bind_compute_pipeline(&self.collision_detection)
            .dispatch_1d(self.particles().particles().len() as u32 / 32)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(self.particles().particles().gpu()),
                make_shader_buffer_barrier(&self.constraints_buffer)
            ])
            .bind_compute_pipeline(&self.solve_constraints)
            .dispatch_indirect(&self.indirect_buffer, 0);
        Ok(())
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, player: &Player) {
        self.global.update_uniforms(image_idx, player);
        let throw_uniform = &mut self.particle_uniform.as_slice_mut()[0];
        throw_uniform.position.0 = player.location().clone();
        throw_uniform.velocity.0 = player.throw_velocity().clone();
        // println!("{:?}", throw_uniform);
    }

    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.global.recreate(render_pass)
    }
}