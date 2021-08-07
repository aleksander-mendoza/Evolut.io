use crate::render::stage_buffer::{StageBuffer, IndirectDispatchBuffer, IndirectDispatchOwnedBuffer, StageOwnedBuffer};
use crate::pipelines::particle::Particle;
use crate::render::owned_buffer::{OwnedBuffer};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Fragment, Vertex, Compute};
use ash::vk::ShaderStageFlags;
use crate::render::pipeline::{PipelineBuilder, BufferBinding, Pipeline};
use ash::vk;
use crate::pipelines::renderable::{RenderResources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::render::submitter::Submitter;
use crate::pipelines::joint::{Joint, JointResources};
use crate::pipelines::particles::{ParticleResources, Particles};
use crate::pipelines::block_world::{BlockWorldResources, BlockWorld};
use crate::render::compute::{ComputePipelineBuilder, ComputePipeline, StorageBufferBinding, UniformBufferBinding};
use crate::blocks::world_size::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, CHUNK_WIDTH_IN_CELLS, CHUNK_DEPTH_IN_CELLS, CHUNK_HEIGHT_IN_CELLS};
use crate::render::vector::Vector;
use crate::render::host_buffer::HostBuffer;
use crate::pipelines::player::Player;
use crate::render::uniform_types::Vec3;
use crate::render::buffer_type::{Storage, Cpu, Uniform, GpuIndirect};
use crate::render::buffer::{make_shader_buffer_barrier, Buffer};
use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::pipelines::physics::{PhysicsResources, Physics};
use crate::pipelines::computable::{Computable, ComputeResources};

pub struct GameResources {
    res: JointResources<BlockWorldResources,ParticleResources>,
    physics:PhysicsResources
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
pub struct ThrowUniform {
    position: Vec3,
    velocity: Vec3,
}

impl GameResources {

    pub fn new(cmd_pool: &CommandPool, foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        let particles = ParticleResources::new(cmd_pool, foundations)?;
        let world = BlockWorldResources::new(cmd_pool, foundations)?;
        let res = JointResources::new(world, particles);
        let physics = PhysicsResources::new(cmd_pool,foundations)?;
        Ok(Self { res , physics})
    }
}

impl RenderResources for GameResources {
    type Render = Game;
    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder, foundations:&FoundationInitializer) -> Result<(), Error> {
        self.res.create_descriptors(descriptors,foundations)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, Error> {
        let Self { res, physics } = self;
        let global = res.make_renderable(cmd_pool, render_pass, descriptors, foundations)?;
        let physics = physics.make_computable(cmd_pool,foundations)?;
        Ok(Game {
            physics,
            global,
        })
    }
}


pub struct Game {
    global: Joint<BlockWorld, Particles>,
    physics: Physics,
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
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors: &Descriptors, render_pass: &SingleRenderPass, foundations:&Foundations) -> Result<(), Error> {
        self.global.record_cmd_buffer(cmd, image_idx, descriptors, render_pass, foundations)
    }

    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations:&Foundations) -> Result<(), Error> {
        self.global.record_compute_cmd_buffer(cmd,foundations)?;
        self.physics.record_compute_cmd_buffer(cmd,foundations)
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, player: &Player) {
        self.global.update_uniforms(image_idx, player);
        self.physics.update_uniforms(player);
    }

    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.global.recreate(render_pass)
    }
}