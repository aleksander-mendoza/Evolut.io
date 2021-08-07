
use crate::render::stage_buffer::StageBuffer;
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::ShaderModule;
use ash::vk::ShaderStageFlags;
use crate::render::pipeline::{PipelineBuilder, BufferBinding, Pipeline};
use ash::vk;
use crate::pipelines::renderable::{RenderResources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::pipelines::particles::ParticleResources;
use crate::pipelines::block_world::BlockWorldResources;
use crate::pipelines::player::Player;
use crate::pipelines::foundations::{Foundations, FoundationInitializer};

pub struct JointResources<A: RenderResources,B: RenderResources> {
    a: A,
    b: B
}

impl <A: RenderResources,B: RenderResources> JointResources<A,B>{
    pub fn a(&self)->&A{
        &self.a
    }
    pub fn b(&self)->&B{
        &self.b
    }
    pub fn a_mut(&mut self)->&mut A{
        &mut self.a
    }
    pub fn b_mut(&mut self)->&mut B{
        &mut self.b
    }
}
impl <A: RenderResources,B: RenderResources> JointResources<A,B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}
impl <A: RenderResources,B: RenderResources> RenderResources for JointResources<A,B> {
    type Render = Joint<A::Render,B::Render>;

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder, foundations:&FoundationInitializer) -> Result<(), Error> {
        self.a.create_descriptors(descriptors,foundations)?;
        self.b.create_descriptors(descriptors,foundations)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, Error> {
        let Self{a,b} = self;
        Ok(Joint{ a:a.make_renderable(cmd_pool,render_pass,descriptors,foundations)?, b:b.make_renderable(cmd_pool,render_pass,descriptors,foundations)? })
    }
}


pub struct Joint<A:Renderable,B:Renderable> {
    a:A,
    b:B
}

impl <A:Renderable,B:Renderable> Joint<A,B>{
    pub fn a(&self)->&A{
        &self.a
    }
    pub fn b(&self)->&B{
        &self.b
    }
    pub fn a_mut(&mut self)->&mut A{
        &mut self.a
    }
    pub fn b_mut(&mut self)->&mut B{
        &mut self.b
    }
}

impl <A:Renderable,B:Renderable> Renderable for Joint<A,B> {


    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors:&Descriptors, render_pass: &SingleRenderPass, foundations:&Foundations) -> Result<(), Error> {
        self.a.record_cmd_buffer(cmd,image_idx,descriptors,render_pass, foundations)?;
        self.b.record_cmd_buffer(cmd,image_idx,descriptors,render_pass, foundations)
    }
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations:&Foundations) -> Result<(), Error> {
        self.a.record_compute_cmd_buffer(cmd,foundations)?;
        self.b.record_compute_cmd_buffer(cmd,foundations)
    }
    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, player:&Player) {
        self.a.update_uniforms(image_idx, player);
        self.b.update_uniforms(image_idx, player);
    }

    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.a.recreate(render_pass)?;
        self.b.recreate(render_pass)
    }
}