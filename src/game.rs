use crate::render::stage_buffer::StageBuffer;
use crate::particle::Particle;
use crate::render::buffer::{Buffer, Cpu, Gpu, Storage, ProceduralStorage};
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
use crate::blocks::world_size::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, CHUNK_WIDTH_IN_CELLS, CHUNK_DEPTH_IN_CELLS, CHUNK_HEIGHT_IN_CELLS};
use crate::render::vector::Vector;

pub struct GameResources {
    res:JointResources<BlockWorldResources,ParticleResources>,
    comp_shader:ShaderModule<Compute>,
    collision_grid:Submitter<Buffer<u32, Storage>>,
    previous_positions:Submitter<StageBuffer<glm::Vec3, Cpu, Storage>>
}

impl Resources for GameResources {
    type Render = Game;

    fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let res = JointResources::<BlockWorldResources,ParticleResources>::new(cmd_pool)?;
        let comp_shader = ShaderModule::new(include_glsl!("assets/shaders/wind.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let mut collision_grid = Submitter::new(Buffer::with_capacity(cmd_pool.device(),CHUNK_WIDTH_IN_CELLS*CHUNK_HEIGHT_IN_CELLS*CHUNK_DEPTH_IN_CELLS)?, cmd_pool)?;
        collision_grid.fill_zeros_submit()?;
        let shifted_particles:Vec<glm::Vec3> = res.b().particles().as_slice().iter().map(|w|Particle::rand_vec3()*0.1).collect();
        let mut previous_positions = StageBuffer::new(cmd_pool, &shifted_particles)?;
        Ok(Self{res, comp_shader, collision_grid, previous_positions})
    }

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder) -> Result<(), Error> {
        self.res.create_descriptors(descriptors)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked) -> Result<Self::Render, Error> {
        let Self{ res, comp_shader, collision_grid, previous_positions } = self;
        let global = res.make_renderable(cmd_pool,render_pass,descriptors)?;
        let mut compute_pipeline = ComputePipelineBuilder::new();
        compute_pipeline.shader("main",comp_shader);
        let particles_binding = compute_pipeline.storage_buffer(global.b().particles().gpu());
        let prev_pos_binding = compute_pipeline.storage_buffer(previous_positions.gpu());
        let collision_grid_binding = compute_pipeline.storage_buffer(&collision_grid);
        let compute_pipeline = compute_pipeline.build(cmd_pool.device())?;
        let collision_grid = collision_grid.take()?;
        Ok(Game{global, compute_pipeline, particles_binding, prev_pos_binding, collision_grid_binding, collision_grid, previous_positions})
    }
}


pub struct Game {
    //The order matters because of drop!
    particles_binding: StorageBufferBinding<Particle>,
    collision_grid_binding: StorageBufferBinding<u32>,
    prev_pos_binding: StorageBufferBinding<glm::Vec3>,
    compute_pipeline: ComputePipeline,
    global: Joint<BlockWorld,Particles>,
    collision_grid:Buffer<u32, Storage>,
    previous_positions:Submitter<StageBuffer<glm::Vec3, Cpu, Storage>>
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