
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Fragment, Vertex};

use crate::render::pipeline::{PipelineBuilder, BufferBinding, Pipeline};
use ash::vk;
use crate::pipelines::renderable::{RenderResources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::swap_chain::SwapchainImageIdx;

use crate::pipelines::player::Player;





use crate::render::buffer::Buffer;




use crate::pipelines::foundations::{FoundationInitializer, Foundations};

pub struct ParticleResources {
    frag: ShaderModule<Fragment>,
    vert: ShaderModule<Vertex>,
}

impl ParticleResources {
    pub fn new(cmd_pool: &CommandPool, _foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        let frag = ShaderModule::new(include_glsl!("assets/shaders/particles.frag", kind: frag) as &[u32], cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/particles.vert") as &[u32], cmd_pool.device())?;
        Ok(Self { vert, frag })
    }
}

impl RenderResources for ParticleResources {
    type Render = Particles;

    fn create_descriptors(&self, _descriptors: &mut DescriptorsBuilder, _foundations:&FoundationInitializer) -> Result<(), Error> {
        Ok(())
    }

    fn make_renderable(self, _cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, Error> {
        let Self {frag, vert} = self;
        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .fragment_shader("main", frag)
            .vertex_shader("main", vert)
            .depth_test(true)
            .topology(vk::PrimitiveTopology::POINT_LIST)
            .color_blend_attachment_states_disabled();
        let particle_binding = pipeline.vertex_input_from(0, foundations.particles());
        let mut particle_builder = ParticleBuilder {
            pipeline,
            particle_binding,
        };
        let particle_compiled = particle_builder.create_pipeline(render_pass)?;
        Ok(Particles { particle_compiled, particle_builder })
    }
}

pub struct ParticleBuilder {
    pipeline: PipelineBuilder,
    particle_binding: BufferBinding<Particle>,
}

impl ParticleBuilder {
    pub fn create_pipeline(&mut self, render_pass: &SingleRenderPass) -> Result<Pipeline, Error> {
        self.pipeline
            .reset_scissors()
            .scissors(render_pass.swapchain().render_area())
            .reset_viewports()
            .viewports(render_pass.swapchain().viewport())
            .build(render_pass)
    }
}


pub struct Particles {
    particle_compiled: Pipeline,
    particle_builder: ParticleBuilder,
}

impl Particles {
    pub fn pipeline(&self) -> &Pipeline {
        &self.particle_compiled
    }
}

impl Renderable for Particles {
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors: &Descriptors, _render_pass: &SingleRenderPass, foundations:&Foundations) -> Result<(), Error> {
        cmd.bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), descriptors.descriptor_set(image_idx))
            .vertex_input(self.particle_builder.particle_binding, foundations.particles())
            .draw_indirect(foundations.indirect().draw_particles());
        Ok(())
    }
    fn record_compute_cmd_buffer(&self, _cmd: &mut CommandBuffer, _foundations:&Foundations) -> Result<(), Error> {
        Ok(())
    }

    fn update_uniforms(&mut self, _image_idx: SwapchainImageIdx, _player: &mut Player) {}
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.particle_compiled = self.particle_builder.create_pipeline(render_pass)?;
        Ok(())
    }
}