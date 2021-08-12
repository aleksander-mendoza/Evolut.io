use crate::blocks::{Face, Block, WorldBlocks, WorldFaces};
use crate::blocks::block_properties::{BEDROCK, DIRT, GRASS, PLANK, GLASS};
use ash::vk;
use crate::render::shader_module::{ShaderModule, Fragment, Vertex, Compute};

use crate::render::pipeline::{PipelineBuilder, BufferBinding, PushConstant, Pipeline};
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::submitter::Submitter;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use failure::Error;
use crate::render::texture::{StageTexture, TextureView, Dim2D};



use crate::render::descriptors::{Descriptors, DescriptorsBuilder, DescriptorsBuilderLocked};

use crate::render::imageview::Color;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::pipelines::player::Player;

use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::pipelines::renderable::{RenderResources, Renderable};
use crate::render::stage_buffer::{StageBuffer, StageSubBuffer};
use crate::render::buffer_type::{Storage, Cpu};
use crate::render::compute::ComputePipeline;
use crate::render::buffer::Buffer;

pub struct BlockWorldResources {
    texture: Submitter<StageTexture<Dim2D>>,
    frag:ShaderModule<Fragment>,
    vert:ShaderModule<Vertex>,
}
impl BlockWorldResources{
    // pub fn world(&self) -> &Submitter<World>{
    //     &self.world
    // }
    pub fn new(cmd_pool: &CommandPool, foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        let texture = StageTexture::new("assets/img/blocks.png".as_ref(), cmd_pool, true)?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/block.frag", kind: frag) as &[u32], cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/block.vert") as &[u32], cmd_pool.device())?;
        Ok(Self {
            texture,
            frag,
            vert,
        })
    }
}
impl RenderResources for BlockWorldResources{
    type Render = BlockWorld;
    fn create_descriptors(&self,descriptors:&mut DescriptorsBuilder, foundations:&FoundationInitializer)->Result<(),failure::Error>{
        descriptors.sampler(foundations.sampler(), self.texture.imageview());
        Ok(())
    }

    fn make_renderable(self, _cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors:&DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, failure::Error>{
        let Self{ texture,
             frag, vert,
        } = self;
        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .fragment_shader("main", frag)
            .vertex_shader("main", vert)
            .depth_test(true)
            .cull_face(vk::CullModeFlags::FRONT)
            .front_face_clockwise(true)
            .color_blend_attachment_states_default();

        let instance_binding = pipeline.instance_input_from(0, foundations.opaque_and_transparent_face_buffer());
        let texture = texture.take()?.take();
        let mut block_world_builder = BlockWorldBuilder {
            pipeline,
            texture,
            instance_binding,
        };
        let block_world_compiled = block_world_builder.create_pipeline(render_pass)?;
        Ok(BlockWorld { block_world_builder, block_world_compiled })
    }

}

pub struct BlockWorldBuilder {
    pipeline: PipelineBuilder,
    texture: TextureView<Dim2D, Color>,
    instance_binding: BufferBinding<Face>,
}

impl BlockWorldBuilder {

    pub fn create_pipeline(&mut self, render_pass: &SingleRenderPass) -> Result<Pipeline, Error> {
        self.pipeline
            .reset_scissors()
            .scissors(render_pass.swapchain().render_area())
            .reset_viewports()
            .viewports(render_pass.swapchain().viewport())
            .build(render_pass)
    }

}

pub struct BlockWorld {
    block_world_compiled: Pipeline,
    block_world_builder: BlockWorldBuilder,
}

impl BlockWorld {

    // pub fn world_mut(&mut self) -> &mut Submitter<World> {
    //     &mut self.block_world_builder.world
    // }
    // pub fn world(&self) -> &Submitter<World> {
    //     &self.block_world_builder.world
    // }
    pub fn pipeline(&self) -> &Pipeline {
        &self.block_world_compiled
    }

}

impl Renderable for BlockWorld {


    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors:&Descriptors, _render_pass: &SingleRenderPass, foundations:&Foundations) -> Result<(), Error> {
        cmd
            .bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), descriptors.descriptor_set(image_idx))
            .vertex_input(self.block_world_builder.instance_binding,foundations.opaque_and_transparent_face_buffer())
            .draw_indirect(foundations.indirect().draw_blocks());
        Ok(())
    }

    fn record_compute_cmd_buffer(&self, _cmd: &mut CommandBuffer, _foundations:&Foundations) -> Result<(), Error> {
        Ok(())
    }

    fn update_uniforms(&mut self, _image_idx: SwapchainImageIdx, _player:&mut Player) {
    }
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.block_world_compiled = self.block_world_builder.create_pipeline(render_pass)?;
        Ok(())
    }
}

