use crate::render::stage_buffer::StageBuffer;
use crate::particle::Particle;
use crate::render::buffer::{Cpu, Gpu, Storage};
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
use crate::render::submitter::Submitter;
use crate::player::Player;

pub struct ParticleResources {
    particles: Submitter<StageBuffer<Particle, Cpu, Storage>>,
    frag:ShaderModule<Fragment>,
    vert:ShaderModule<Vertex>,
}

impl ParticleResources{
    pub fn particles(&self) -> &StageBuffer<Particle, Cpu, Storage>{
        &self.particles
    }
}

impl Resources for ParticleResources {
    type Render = Particles;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let mut particles_data:Vec<Particle> = std::iter::repeat_with(Particle::random).take(512).collect();
        particles_data[1].new_position = glm::vec3(2.,7.,2.);
        particles_data[1].old_position = particles_data[1].new_position;
        particles_data[2].new_position = particles_data[1].new_position+glm::vec3(0.3,0.,0.);
        particles_data[2].old_position = particles_data[2].new_position;
        particles_data[3].new_position = particles_data[1].new_position+glm::vec3(0.3,0.3,0.);
        particles_data[3].old_position = particles_data[3].new_position;
        particles_data[4].new_position = particles_data[1].new_position+glm::vec3(0.,0.3,0.);
        particles_data[4].old_position = particles_data[4].new_position;
        let particles = StageBuffer::new(cmd_pool, &particles_data)?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/particles.frag", kind: frag) as &[u32],  cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/particles.vert") as &[u32],  cmd_pool.device())?;
        Ok(Self { particles,vert,frag })
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self{ particles, frag, vert } = self;
        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .fragment_shader("main", frag)
            .vertex_shader("main", vert)
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
        let particles = particles.take()?;
        let particle_binding = pipeline.vertex_input_from(0,particles.gpu());
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
    particles: StageBuffer<Particle, Cpu, Storage>,
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
    pub fn particles(&self) -> &StageBuffer<Particle, Cpu, Storage>{
        &self.particle_builder.particles
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
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer) -> Result<(), Error> {
        Ok(())
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, player:&Player) {
    }
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.particle_compiled = self.particle_builder.create_pipeline(render_pass)?;
        Ok(())
    }
}