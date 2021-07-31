use crate::render::stage_buffer::StageBuffer;
use crate::particle::Particle;
use crate::render::buffer::{Cpu, Gpu};
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
use crate::render::compute::{ComputePipelineBuilder, ComputePipeline, StorageBufferBinding};

pub struct GameResources {
    res:JointResources<BlockWorldResources,ParticleResources>,
    comp_shader:ShaderModule<Compute>
}

impl Resources for GameResources {
    type Render = Game;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let res = JointResources::new(cmd_pool)?;
        let comp_shader = ShaderModule::new(include_glsl!("assets/shaders/wind.comp", kind: comp) as &[u32], cmd_pool.device())?;
        Ok(Self{res, comp_shader})
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        self.res.create_descriptors(descriptors)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self{ res, comp_shader } = self;
        let global = res.make_renderable(cmd_pool,render_pass,descriptors)?;
        let mut compute_pipeline = ComputePipelineBuilder::new();
        compute_pipeline.shader("main",comp_shader);
        let particles_binding = compute_pipeline.storage_buffer(global.b().particles().gpu());
        let compute_pipeline = compute_pipeline.build(cmd_pool.device())?;
        Ok(Game{global, compute_pipeline, particles_binding})
    }
}


pub struct Game {
    //The order matters because of drop!
    particles_binding: StorageBufferBinding<Particle>,
    compute_pipeline: ComputePipeline,
    global: Joint<BlockWorld,Particles>,
}

impl Game {
    pub fn block_world(&self) -> &BlockWorld {
        self.global.a()
    }
    pub fn particles(&self) -> &Particles{
        self.global.b()
    }
    pub fn block_world_mut(&mut self) -> &mut BlockWorld {
        self.global.a_mut()
    }
    pub fn particles_mut(&mut self) -> &mut Particles{
        self.global.b_mut()
    }
}

impl Renderable for Game {


    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors:&Descriptors, render_pass: &SingleRenderPass) -> Result<(), Error> {

        self.global.record_cmd_buffer(cmd,image_idx,descriptors,render_pass)?;
        Ok(())
    }

    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer) -> Result<(), Error> {
        cmd.bind_compute_pipeline(&self.compute_pipeline)
            .dispatch_1d(self.particles().particles().len() as u32);
        Ok(())
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx) {
        self.global.update_uniforms(image_idx)
    }

    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.global.recreate(render_pass)
    }
}