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

pub struct ParticleResources {
    particles: StageBuffer<Particle, Cpu, Gpu>,
    frag:ShaderModule,
    vert:ShaderModule,
}

impl Resources for ParticleResources {
    type Render = Particles;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let particles_data:Vec<Particle> = std::iter::repeat_with(Particle::random).take(64).collect();
        let particles = StageBuffer::new(cmd_pool, &particles_data)?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/particles.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/particles.vert") as &[u32], ShaderStageFlags::VERTEX, cmd_pool.device())?;
        let particles = particles.take()?;
        Ok(Self { particles,vert,frag })
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self{ particles, frag, vert } = self;
        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .shader("main", frag)
            .shader("main", vert)
            .depth_test(true)
            .topology(vk::PrimitiveTopology::POINT_LIST)
            .color_blend_attachment_states(vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::all(),
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ZERO,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            });
        let particle_binding = pipeline.vertex_input::<Particle>(0);
        let mut particle_builder = ParticleBuilder {
            particles,
            pipeline,
            particle_binding
        };
        let particle_compiled = particle_builder.create_pipeline(render_pass)?;
        Ok(Particles{ particle_compiled, particle_builder })
    }
}


pub struct ParticleBuilder {
    particles: StageBuffer<Particle, Cpu, Gpu>,
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


    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors:&Descriptors, render_pass: &SingleRenderPass) -> Result<(), Error> {
        cmd.bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), descriptors.descriptor_set(image_idx))
            .vertex_input(self.particle_builder.particle_binding, self.particle_builder.particles.gpu())
            .draw(self.particle_builder.particles.len() as u32,1,0,0);
        Ok(())
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx) {
    }
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.particle_compiled = self.particle_builder.create_pipeline(render_pass)?;
        Ok(())
    }
}