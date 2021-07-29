use crate::blocks::{World, Face};
use crate::blocks::block_properties::{BEDROCK, DIRT, GRASS, PLANK, GLASS};
use ash::vk;
use crate::render::shader_module::ShaderModule;
use ash::vk::{ShaderStageFlags, PFN_vkCmdResetQueryPool};
use crate::render::pipeline::{PipelineBuilder, BufferBinding, PushConstant, Pipeline};
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::submitter::Submitter;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use failure::Error;
use crate::render::texture::{StageTexture, TextureView, Dim2D};
use crate::render::sampler::Sampler;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::device::Device;
use crate::render::descriptors::{Descriptors, DescriptorsBuilder, DescriptorsBuilderLocked, UniformBufferBinding};
use crate::mvp_uniforms::MvpUniforms;
use crate::render::imageview::Color;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::display::Renderable;

pub struct BlockWorldBuilder {
    pipeline: PipelineBuilder,
    texture: TextureView<Dim2D, Color>,
    sampler: Sampler,
    world: Submitter<World>,
    instance_binding: BufferBinding<Face>,
    chunk_position_binding: PushConstant<glm::Vec3>,
    descriptors: DescriptorsBuilderLocked,
    mvp_uniforms_binding: UniformBufferBinding<MvpUniforms, 1>,
}

impl BlockWorldBuilder {
    pub fn new(cmd_pool: &CommandPool, mvp_uniforms: &MvpUniforms) -> Result<Self, failure::Error> {
        let texture = StageTexture::new("assets/img/blocks.png".as_ref(), cmd_pool, true)?;
        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/block.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/block.vert") as &[u32], ShaderStageFlags::VERTEX, cmd_pool.device())?;
        let mut world = World::new(1,1, cmd_pool)?;
        world.blocks_mut().no_update_fill_level(0, 1, BEDROCK);
        world.blocks_mut().no_update_fill_level(1, 1, DIRT);
        world.blocks_mut().no_update_fill_level(2, 1, GRASS);
        world.blocks_mut().no_update_fill_level(10, 1, GLASS);
        // world.blocks_mut().no_update_outline(5, 2, 5, 5, 5, 5, PLANK);
        world.compute_faces();
        world.flush_all_chunks()?;
        let texture = texture.take()?.take();
        let mut descriptors = DescriptorsBuilder::new();
        descriptors.sampler(&sampler, texture.imageview());
        let mvp_uniforms_binding = descriptors.singleton_uniform_buffer(mvp_uniforms);
        let descriptors = descriptors.make_layout(cmd_pool.device())?;

        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .shader("main", frag)
            .shader("main", vert)
            .depth_test(true)
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
        let instance_binding = pipeline.instance_input::<Face>(0);
        let chunk_position_binding = pipeline.push_constant::<glm::Vec3>();

        Ok(Self {
            mvp_uniforms_binding,
            pipeline,
            texture,
            sampler,
            world,
            instance_binding,
            chunk_position_binding,
            descriptors,
        })
    }

    pub fn create_pipeline(&mut self, render_pass: &SingleRenderPass) -> Result<BlockWorldPipeline, Error> {
        let descriptors = self.descriptors.build(render_pass.swapchain())?;
        let pipeline = self.pipeline
            .reset_scissors()
            .scissors(render_pass.swapchain().render_area())
            .reset_viewports()
            .viewports(render_pass.swapchain().viewport())
            .build(render_pass)?;
        Ok(BlockWorldPipeline { descriptors, pipeline })
    }
}

pub struct BlockWorldPipeline {
    descriptors: Descriptors,
    pipeline: Pipeline,
}

pub struct BlockWorld {
    block_world_compiled: BlockWorldPipeline,
    block_world: BlockWorldBuilder,
}

impl BlockWorld {
    const CLEAR_VALUES: [vk::ClearValue; 2] = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }, vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.,
            stencil: 0,
        },
    }];

    pub fn world_mut(&mut self) -> &mut Submitter<World> {
        &mut self.block_world.world
    }
    pub fn world(&self) -> &Submitter<World> {
        &self.block_world.world
    }
    pub fn pipeline(&self) -> &Pipeline {
        &self.block_world_compiled.pipeline
    }
    pub fn descriptors(&self) -> &Descriptors {
        &self.block_world_compiled.descriptors
    }
}

impl Renderable<MvpUniforms> for BlockWorld {
    fn new(cmd_pool: &CommandPool, render_pass: &SingleRenderPass, mvp_uniforms: &MvpUniforms) -> Result<Self, Error> {
        let mut block_world = BlockWorldBuilder::new(cmd_pool, mvp_uniforms)?;
        let block_world_compiled = block_world.create_pipeline(render_pass)?;
        Ok(Self { block_world, block_world_compiled })
    }

    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, render_pass: &SingleRenderPass) -> Result<(), Error> {
        cmd
            .reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?
            .render_pass(render_pass, render_pass.framebuffer(image_idx), render_pass.swapchain().render_area(), &Self::CLEAR_VALUES)
            .bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), self.descriptors().descriptor_set(image_idx));
        self.block_world.world.draw(cmd, self.pipeline(),self.block_world.instance_binding,self.block_world.chunk_position_binding);
        cmd
            .end_render_pass()
            .end()?;
        Ok(())
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, mvp_uniform: &MvpUniforms) {
        self.block_world_compiled.descriptors.uniform_as_slice_mut(image_idx, self.block_world.mvp_uniforms_binding).copy_from_slice(std::slice::from_ref(mvp_uniform))
    }
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.block_world_compiled = self.block_world.create_pipeline(render_pass)?;
        Ok(())
    }
}

