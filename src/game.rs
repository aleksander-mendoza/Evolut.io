use crate::render::stage_buffer::{StageBuffer, IndirectDispatchBuffer, IndirectDispatchOwnedBuffer, StageOwnedBuffer};
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
use crate::render::buffer::{make_shader_buffer_barrier, Buffer};

pub struct GameResources {
    res: JointResources<BlockWorldResources, ParticleResources>,
    collision_detection: ShaderModule<Compute>,
    solve_constraints: ShaderModule<Compute>,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
}



#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
pub struct ThrowUniform {
    position: Vec3,
    velocity: Vec3,
}

impl GameResources {

    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let world = BlockWorldResources::new(cmd_pool)?;
        let particles = ParticleResources::new(cmd_pool, world.world().size())?;
        let res = JointResources::new(world, particles);
        let collision_detection = ShaderModule::new(include_glsl!("assets/shaders/collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let solve_constraints = ShaderModule::new(include_glsl!("assets/shaders/solve_constraints.comp", kind: comp) as &[u32], cmd_pool.device())?;

        let world_size = res.a().world().size();

        let particle_uniform = HostBuffer::new(cmd_pool.device(), &[ThrowUniform {
            position: Vec3(glm::vec3(0., 0., 0.)),
            velocity: Vec3(glm::vec3(0., 0., 0.)),
        }])?;

        Ok(Self { res, collision_detection, solve_constraints, particle_uniform })
    }
}

impl Resources for GameResources {
    type Render = Game;
    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        self.res.create_descriptors(descriptors)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self {
            res,
            collision_detection,
            solve_constraints,
            particle_uniform,
        } = self;
        let global = res.make_renderable(cmd_pool, render_pass, descriptors)?;
        let mut compute_pipeline = ComputePipelineBuilder::new();
        let uniform_binding = compute_pipeline.uniform_buffer(particle_uniform.buffer());
        compute_pipeline.storage_buffer(global.b().constants());
        compute_pipeline.storage_buffer(global.b().particles());
        compute_pipeline.storage_buffer(global.b().collision_grid());
        compute_pipeline.storage_buffer(global.b().constraints());
        compute_pipeline.storage_buffer(global.b().indirect());
        compute_pipeline.shader("main", collision_detection);
        let collision_detection = compute_pipeline.build(cmd_pool.device())?;
        compute_pipeline.shader("main", solve_constraints);
        let solve_constraints = compute_pipeline.build(cmd_pool.device())?;
        Ok(Game {
            global,
            collision_detection,
            solve_constraints,
            uniform_binding,
            particle_uniform,
        })
    }
}


pub struct Game {
    //The order matters because of drop!
    uniform_binding: UniformBufferBinding<ThrowUniform>,
    collision_detection: ComputePipeline,
    solve_constraints: ComputePipeline,
    global: Joint<BlockWorld, Particles>,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
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
            .dispatch_1d(self.particles().particles().bytes() as u32 / 32)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(self.particles().particles()),
                make_shader_buffer_barrier(self.particles().constraints())
            ])
            .bind_compute_pipeline(&self.solve_constraints)
            .dispatch_indirect(self.global.b().indirect(), 0);
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