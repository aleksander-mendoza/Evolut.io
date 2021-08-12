use crate::pipelines::renderable::{RenderResources, Renderable};
use crate::render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use crate::render::single_render_pass::SingleRenderPass;
use failure::Error;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Fragment, Vertex};

use crate::render::submitter::Submitter;
use crate::render::texture::{StageTexture, Dim2D, TextureView};

use ash::vk;
use crate::render::pipeline::{PipelineBuilder, Pipeline, BufferBinding};
use crate::pipelines::bone::Bone;
use crate::render::swap_chain::SwapchainImageIdx;
use crate::pipelines::player::Player;
use crate::render::imageview::Color;


use crate::pipelines::foundations::{Foundations, FoundationInitializer};
use crate::render::buffer::Buffer;

pub struct BoneResources{
    texture: Submitter<StageTexture<Dim2D>>,
    frag:ShaderModule<Fragment>,
    vert:ShaderModule<Vertex>,
}

impl BoneResources{
    pub fn new(cmd_pool: &CommandPool, _foundations:&FoundationInitializer)->Result<Self, failure::Error> {
        let texture = StageTexture::new("assets/img/mobs.jpeg".as_ref(), cmd_pool, true)?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/bones.frag", kind: frag) as &[u32], cmd_pool.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/bones.vert") as &[u32], cmd_pool.device())?;
        Ok(Self{frag,vert,texture})
    }
}

impl RenderResources for BoneResources {
    type Render = Bones;

    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder, foundations:&FoundationInitializer) -> Result<(), Error> {
        descriptors.storage_buffer(foundations.particles().gpu());
        descriptors.sampler(foundations.sampler(), self.texture.imageview());
        Ok(())
    }

    fn make_renderable(self, _cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, Error> {
        let Self{ texture, frag, vert } = self;
        let mut pipeline = PipelineBuilder::new();
        pipeline.descriptor_layout(descriptors.layout().clone())
            .fragment_shader("main", frag)
            .vertex_shader("main", vert)
            .depth_test(true)
            .cull_face(vk::CullModeFlags::FRONT)
            .front_face_clockwise(false)
            .color_blend_attachment_states_disabled();
        let instance_binding = pipeline.instance_input_from(0,foundations.bones());

        let texture = texture.take()?.take();
        let mut bones_builder = BonesBuilder {
            pipeline,
            texture,
            instance_binding,
        };
        let bones_compiled = bones_builder.create_pipeline(render_pass)?;
        Ok(Bones { bones_builder, bones_compiled })
    }
}


pub struct BonesBuilder {
    pipeline: PipelineBuilder,
    texture: TextureView<Dim2D, Color>,
    instance_binding: BufferBinding<Bone>,
}

impl BonesBuilder{
    pub fn create_pipeline(&mut self, render_pass: &SingleRenderPass) -> Result<Pipeline, Error> {
        self.pipeline
            .reset_scissors()
            .scissors(render_pass.swapchain().render_area())
            .reset_viewports()
            .viewports(render_pass.swapchain().viewport())
            .build(render_pass)
    }
}


pub struct Bones {
    bones_compiled: Pipeline,
    bones_builder: BonesBuilder,
}

impl Bones {

    pub fn pipeline(&self) -> &Pipeline {
        &self.bones_compiled
    }

}

impl Renderable for Bones {


    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors:&Descriptors, _render_pass: &SingleRenderPass, foundations:&Foundations) -> Result<(), Error> {
        cmd
            .bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), descriptors.descriptor_set(image_idx))
            .vertex_input(self.bones_builder.instance_binding,foundations.bones())
            .draw_indirect( foundations.indirect().draw_bones());

        Ok(())
    }

    fn record_compute_cmd_buffer(&self, _cmd: &mut CommandBuffer, _foundations:&Foundations) -> Result<(), Error> {
        Ok(())
    }

    fn update_uniforms(&mut self, _image_idx: SwapchainImageIdx, _player:&mut Player) {
    }

    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error> {
        self.bones_compiled = self.bones_builder.create_pipeline(render_pass)?;
        Ok(())
    }
}
