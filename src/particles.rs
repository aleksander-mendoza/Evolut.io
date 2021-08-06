use crate::render::stage_buffer::{StageBuffer, StageSubBuffer};
use crate::particle::Particle;
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
use crate::render::submitter::{Submitter, fill_submit};
use crate::player::Player;
use crate::render::buffer_type::{Cpu, Storage};
use crate::render::owned_buffer::OwnedBuffer;
use crate::blocks::world_size::CHUNK_VOLUME_IN_CELLS;
use crate::render::subbuffer::SubBuffer;
use crate::constraint::Constraint;
use crate::render::buffer::Buffer;

pub struct ParticleResources {
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    collision_grid: Submitter<SubBuffer<u32, Storage>>,
    constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    frag: ShaderModule<Fragment>,
    vert: ShaderModule<Vertex>,
}

impl ParticleResources {
    pub fn particles(&self) -> &StageSubBuffer<Particle, Cpu, Storage> {
        &self.particles
    }
    pub fn constraints(&self) -> &StageSubBuffer<Constraint, Cpu, Storage> {
        &self.constraints
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}

impl Resources for ParticleResources {
    type Render = Particles;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let particles = 512;
        let max_constraints = 128;
        let grid_size = CHUNK_VOLUME_IN_CELLS as u64;
        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * particles;
        let grid_in_bytes = std::mem::size_of::<u32>() as u64 * grid_size;
        let constraints_in_bytes = std::mem::size_of::<Constraint>() as u64 * max_constraints;
        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            particles_in_bytes +
                                                                                grid_in_bytes +
                                                                                constraints_in_bytes)?;

        let particle_buffer = super_buffer.sub(0..particles_in_bytes).reinterpret_into::<Particle>();
        let grid_buffer = super_buffer.sub(particles_in_bytes..(particles_in_bytes + grid_in_bytes)).reinterpret_into::<u32>();
        let constraint_buffer = super_buffer.sub((particles_in_bytes + grid_in_bytes)..(particles_in_bytes + grid_in_bytes + constraints_in_bytes)).reinterpret_into::<Constraint>();

        let mut particles_data: Vec<Particle> = std::iter::repeat_with(Particle::random).take(particles as usize).collect();
        particles_data[1].new_position = glm::vec3(2., 7., 2.);
        particles_data[1].old_position = particles_data[1].new_position;
        particles_data[2].new_position = particles_data[1].new_position + glm::vec3(0.3, 0., 0.);
        particles_data[2].old_position = particles_data[2].new_position;
        particles_data[3].new_position = particles_data[1].new_position + glm::vec3(0.3, 0.3, 0.);
        particles_data[3].old_position = particles_data[3].new_position;
        particles_data[4].new_position = particles_data[1].new_position + glm::vec3(0., 0.3, 0.);
        particles_data[4].old_position = particles_data[4].new_position;
        let particles = StageBuffer::wrap(cmd_pool, &particles_data, particle_buffer)?;

        let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        fill_submit(&mut collision_grid,u32::MAX)?;

        let d = 0.4;
        let constraints = StageBuffer::wrap(cmd_pool, &[
            Constraint::distance(1, 2, d),
            Constraint::distance(2, 3, d),
            Constraint::distance(3, 4, d),
            Constraint::distance(4, 1, d),
            Constraint::distance(4, 2, d * 2f32.sqrt()),
        ], constraint_buffer)?;

        let frag = ShaderModule::new(include_glsl!("assets/shaders/particles.frag", kind: frag) as &[u32], cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/particles.vert") as &[u32], cmd_pool.device())?;
        Ok(Self { particles, constraints, collision_grid, vert, frag })
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self {
            particles,
            collision_grid,
            constraints,
            frag, vert
        } = self;
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
        let particles = particles.take()?.take_gpu();
        let collision_grid = collision_grid.take()?;
        let constraints = constraints.take()?.take_gpu();
        let particle_binding = pipeline.vertex_input_from(0, &particles);
        let mut particle_builder = ParticleBuilder {
            collision_grid,
            constraints,
            particles,
            pipeline,
            particle_binding,
        };
        let particle_compiled = particle_builder.create_pipeline(render_pass)?;
        Ok(Particles { particle_compiled, particle_builder })
    }
}


pub struct ParticleBuilder {
    particles: SubBuffer<Particle, Storage>,
    constraints: SubBuffer<Constraint, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
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
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particle_builder.particles
    }

    pub fn constraints(&self) -> &SubBuffer<Constraint, Storage> {
        &self.particle_builder.constraints
    }

    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.particle_builder.collision_grid
    }
}

impl Renderable for Particles {
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors: &Descriptors, render_pass: &SingleRenderPass) -> Result<(), Error> {
        cmd.bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), descriptors.descriptor_set(image_idx))
            .vertex_input(self.particle_builder.particle_binding, self.particles())
            .draw(self.particles().len() as u32, 1, 0, 0);
        Ok(())
    }
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer) -> Result<(), Error> {
        Ok(())
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, player: &Player) {}
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.particle_compiled = self.particle_builder.create_pipeline(render_pass)?;
        Ok(())
    }
}