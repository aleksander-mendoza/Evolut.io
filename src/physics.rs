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
use crate::foundations::Foundations;

pub struct PhysicsInitializer {
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

impl PhysicsInitializer {

    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let collision_detection = ShaderModule::new(include_glsl!("assets/shaders/collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let solve_constraints = ShaderModule::new(include_glsl!("assets/shaders/solve_constraints.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let particle_uniform = HostBuffer::new(cmd_pool.device(), &[ThrowUniform {
            position: Vec3(glm::vec3(0., 0., 0.)),
            velocity: Vec3(glm::vec3(0., 0., 0.)),
        }])?;
        Ok(Self {  collision_detection, solve_constraints, particle_uniform })
    }

    pub fn build(self, cmd_pool: &CommandPool, foundations:&Foundations) -> Result<Physics, Error> {
        let Self {
            collision_detection,
            solve_constraints,
            particle_uniform,
        } = self;
        let mut compute_pipeline = ComputePipelineBuilder::new();
        let uniform_binding = compute_pipeline.uniform_buffer(particle_uniform.buffer());
        compute_pipeline.storage_buffer(foundations.constants());
        compute_pipeline.storage_buffer(foundations.particles());
        compute_pipeline.storage_buffer(foundations.collision_grid());
        compute_pipeline.storage_buffer(foundations.constraints());
        compute_pipeline.storage_buffer(foundations.indirect());
        compute_pipeline.shader("main", collision_detection);
        let collision_detection = compute_pipeline.build(cmd_pool.device())?;
        compute_pipeline.shader("main", solve_constraints);
        let solve_constraints = compute_pipeline.build(cmd_pool.device())?;
        Ok(Physics {
            collision_detection,
            solve_constraints,
            uniform_binding,
            particle_uniform,
        })
    }
}


pub struct Physics {
    //The order matters because of drop!
    uniform_binding: UniformBufferBinding<ThrowUniform>,
    collision_detection: ComputePipeline,
    solve_constraints: ComputePipeline,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
}

impl Physics {
    pub fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer,foundations:&Foundations) -> Result<(), Error> {
        cmd.bind_compute_pipeline(&self.collision_detection)
            .dispatch_1d(foundations.particles().bytes() as u32 / 32)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.particles()),
                make_shader_buffer_barrier(foundations.constraints())
            ])
            .bind_compute_pipeline(&self.solve_constraints)
            .dispatch_indirect(foundations.indirect(), 0);
        Ok(())
    }

    pub fn update_player(&mut self, player: &Player) {
        let throw_uniform = &mut self.particle_uniform.as_slice_mut()[0];
        throw_uniform.position.0 = player.location().clone();
        throw_uniform.velocity.0 = player.throw_velocity().clone();
    }

}