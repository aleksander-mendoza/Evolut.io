use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, StageOwnedBuffer, IndirectDispatchSubBuffer};
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
use crate::render::buffer_type::{Cpu, Storage, GpuIndirect};
use crate::render::owned_buffer::OwnedBuffer;
use crate::blocks::world_size::CHUNK_VOLUME_IN_CELLS;
use crate::render::subbuffer::SubBuffer;
use crate::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::particle_constants::ParticleConstants;
use crate::blocks::WorldSize;

pub struct ParticleResources {
    particles: Submitter<StageSubBuffer<Particle, Cpu, Storage>>,
    collision_grid: Submitter<SubBuffer<u32, Storage>>,
    constraints: Submitter<StageSubBuffer<Constraint, Cpu, Storage>>,
    particle_constants: Submitter<StageSubBuffer<ParticleConstants, Cpu, Storage>>,
    indirect: Submitter<IndirectDispatchSubBuffer>,
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

impl ParticleResources {
    pub fn new(cmd_pool: &CommandPool, world_size:&WorldSize) -> Result<Self, failure::Error> {
        let particles = 512u64;
        let max_constraints = 128u64;
        let grid_size = CHUNK_VOLUME_IN_CELLS as u64;
        let solid_particles = 256;
        let phantom_particles = 256;
        debug_assert!(solid_particles + phantom_particles <= particles);

        let mut particles_data: Vec<Particle> = std::iter::repeat_with(Particle::random).take(particles as usize).collect();
        particles_data[1].new_position = glm::vec3(2., 7., 2.);
        particles_data[1].old_position = particles_data[1].new_position;
        particles_data[2].new_position = particles_data[1].new_position + glm::vec3(0.3, 0., 0.);
        particles_data[2].old_position = particles_data[2].new_position;
        particles_data[3].new_position = particles_data[1].new_position + glm::vec3(0.3, 0.3, 0.);
        particles_data[3].old_position = particles_data[3].new_position;
        particles_data[4].new_position = particles_data[1].new_position + glm::vec3(0., 0.3, 0.);
        particles_data[4].old_position = particles_data[4].new_position;

        let d = 0.4;
        let predefined_constraints = vec![
            Constraint::distance(1, 2, d),
            Constraint::distance(2, 3, d),
            Constraint::distance(3, 4, d),
            Constraint::distance(4, 1, d),
            Constraint::distance(4, 2, d * 2f32.sqrt()),
        ];

        let constants = ParticleConstants {
            predefined_constraints: predefined_constraints.len() as u32,
            collision_constraints: 0,
            solid_particles:solid_particles as u32,
            phantom_particles:phantom_particles as u32,
            chunks_x: world_size.width() as u32,
            chunks_z: world_size.depth() as u32,
        };

        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * particles;
        let grid_in_bytes = std::mem::size_of::<u32>() as u64 * grid_size;
        let constraints_in_bytes = std::mem::size_of::<Constraint>() as u64 * max_constraints;
        let constants_in_bytes = std::mem::size_of_val(&constants) as u64;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            particles_in_bytes +
                                                                                grid_in_bytes +
                                                                                constraints_in_bytes +
                                                                                constants_in_bytes)?;
        let offset = 0;
        let particle_buffer = super_buffer.sub(offset..offset+particles_in_bytes).reinterpret_into::<Particle>();
        let offset = offset+particles_in_bytes;
        let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        let offset = offset+grid_in_bytes;
        let constraint_buffer = super_buffer.sub(offset..offset + constraints_in_bytes).reinterpret_into::<Constraint>();
        let offset = offset+constraints_in_bytes;
        let constants_buffer = super_buffer.sub(offset..offset + constants_in_bytes).reinterpret_into::<ParticleConstants>();
        let offset = offset+constants_in_bytes;

        let particle_constants = StageBuffer::wrap(cmd_pool, &[constants], constants_buffer)?;

        let particles = StageBuffer::wrap(cmd_pool, &particles_data, particle_buffer)?;

        let mut collision_grid = Submitter::new(grid_buffer, cmd_pool)?;
        fill_submit(&mut collision_grid,u32::MAX)?;

        let indirect = StageBuffer::new_indirect_dispatch_buffer(cmd_pool, &[
            vk::DispatchIndirectCommand { // solve_constraints.comp
                x: 0, // number of predefined_constraints + collision_constraints
                y: 1,
                z: 1,
            }, vk::DispatchIndirectCommand { // collision_detection.comp
                x: phantom_particles.max(solid_particles) as u32,
                y: 1,
                z: 1,
            }, vk::DispatchIndirectCommand {
                x: 0,
                y: 1,
                z: 1,
            }
        ])?;

        let constraints = StageBuffer::wrap(cmd_pool, &predefined_constraints, constraint_buffer)?;

        let frag = ShaderModule::new(include_glsl!("assets/shaders/particles.frag", kind: frag) as &[u32], cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/particles.vert") as &[u32], cmd_pool.device())?;
        Ok(Self { particles, constraints, collision_grid, particle_constants, indirect, vert, frag })
    }
}
impl Resources for ParticleResources {
    type Render = Particles;



    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self {
            particles,
            collision_grid,
            constraints,
            particle_constants,
            indirect,
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
        let particle_constants = particle_constants.take()?.take_gpu();
        let indirect = indirect.take()?.take_gpu();
        let particle_binding = pipeline.vertex_input_from(0, &particles);
        let mut particle_builder = ParticleBuilder {
            collision_grid,
            particle_constants,
            constraints,
            particles,
            indirect,
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
    particle_constants: SubBuffer<ParticleConstants, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
    indirect:  SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
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
    pub fn constants(&self) -> &SubBuffer<ParticleConstants, Storage> {
        &self.particle_builder.particle_constants
    }
    pub fn indirect(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.particle_builder.indirect
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
            .draw(self.particles().bytes() as u32, 1, 0, 0);
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