
use crate::render::stage_buffer::StageBuffer;
use crate::particle::Particle;
use crate::render::buffer::{Cpu, Gpu};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::ShaderModule;
use ash::vk::ShaderStageFlags;
use crate::render::pipeline::{PipelineBuilder, BufferBinding, Pipeline};
use ash::vk;
use crate::display::{Resources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::particles::ParticleResources;
use crate::block_world::BlockWorldResources;
use crate::player::Player;

pub struct JointResources<A:Resources,B:Resources> {
    a: A,
    b: B
}

impl <A:Resources,B:Resources> JointResources<A,B>{
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

impl <A:Resources,B:Resources> Resources for JointResources<A,B> {
    type Render = Joint<A::Render,B::Render>;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        Ok(Self { a: A::new(cmd_pool)?, b: B::new(cmd_pool)? })
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        self.a.create_descriptors(descriptors)?;
        self.b.create_descriptors(descriptors)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self{a,b} = self;
        Ok(Joint{ a:a.make_renderable(cmd_pool,render_pass,descriptors)?, b:b.make_renderable(cmd_pool,render_pass,descriptors)? })
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


    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors:&Descriptors, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.a.record_cmd_buffer(cmd,image_idx,descriptors,render_pass)?;
        self.b.record_cmd_buffer(cmd,image_idx,descriptors,render_pass)
    }
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer) -> Result<(), Error> {
        self.a.record_compute_cmd_buffer(cmd)?;
        self.b.record_compute_cmd_buffer(cmd)
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