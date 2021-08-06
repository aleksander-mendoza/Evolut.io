use crate::render::instance::Instance;
use crate::render::device::{pick_physical_device, Device};
use crate::render::pipeline::{PipelineBuilder, Pipeline};
use crate::render::render_pass::{RenderPassBuilder, RenderPass};
use crate::render::shader_module::ShaderModule;
use ash::vk::{ShaderStageFlags, PhysicalDevice, ClearValue, VertexInputAttributeDescription};
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::imageview::{ImageView, Color, Depth};
use crate::render::frames_in_flight::FramesInFlight;
use crate::render::surface::Surface;
use crate::render::swap_chain::{SwapChain, SwapchainImageIdx};
use crate::render::vulkan_context::VulkanContext;
use failure::Error;
use ash::vk;
use crate::render::data::{VertexClr, VertexClrTex, VertexSource};
use crate::render::owned_buffer::{OwnedBuffer};
use crate::render::fence::Fence;
use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::texture::{StageTexture, Dim2D, TextureView};
use crate::render::sampler::Sampler;
use crate::render::framebuffer::Framebuffer;
use crate::blocks::Face;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::descriptors::{Descriptors, DescriptorsBuilder, UniformBufferBinding, DescriptorsBuilderLocked};
use std::marker::PhantomData;
use crate::player::Player;
use crate::mvp_uniforms::MvpUniforms;

pub trait Resources:Sized{
    type Render:Renderable;
    fn new(cmd_pool:&CommandPool)->Result<Self,failure::Error>;
    fn create_descriptors(&self,descriptors:&mut DescriptorsBuilder)->Result<(),failure::Error>;
    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors:&DescriptorsBuilderLocked) -> Result<Self::Render, failure::Error>;

}
pub trait Renderable:Sized{
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx:SwapchainImageIdx,descriptors:&Descriptors, render_pass:&SingleRenderPass)->Result<(),Error>;
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer)->Result<(),Error>;
    fn update_uniforms(&mut self, image_idx:SwapchainImageIdx, player:&Player);
    fn recreate(&mut self, render_pass: &SingleRenderPass, ) -> Result<(), Error>;
}

pub struct Display<P:Resources>{
    command_buffers: Vec<CommandBuffer>,
    pipeline:P::Render,
    render_pass:SingleRenderPass,
    descriptors:Descriptors,
    descriptors_builder:DescriptorsBuilderLocked,
    cmd_pool: CommandPool,
    uniforms_binding:UniformBufferBinding<MvpUniforms,1>,
    vulkan: VulkanContext,
}
impl <P:Resources> Display<P> {
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

    pub fn pipeline(&self) -> &P::Render{
        &self.pipeline
    }
    pub fn pipeline_mut(&mut self) -> &mut P::Render{
        &mut self.pipeline
    }
    pub fn new(vulkan: VulkanContext, player:&Player) -> Result<Self, failure::Error> {
        let render_pass = vulkan.create_single_render_pass()?;
        let cmd_pool = CommandPool::new(vulkan.device(),true)?;
        let mut descriptors_builder = DescriptorsBuilder::new();
        let uniforms_binding = descriptors_builder.singleton_uniform_buffer(player.mvp_uniforms());
        let resources = P::new(&cmd_pool)?;
        resources.create_descriptors(&mut descriptors_builder)?;
        let descriptors_builder = descriptors_builder.make_layout(cmd_pool.device())?;
        let pipeline = resources.make_renderable(&cmd_pool,&render_pass,&descriptors_builder)?;
        let descriptors = descriptors_builder.build(render_pass.swapchain())?;
        let command_buffers = cmd_pool.create_command_buffers(render_pass.framebuffers_len() as u32)?;
        Ok(Self {
            descriptors_builder,
            descriptors,
            command_buffers,
            pipeline,
            render_pass,
            cmd_pool,
            vulkan,
            uniforms_binding
        })
    }
    pub fn cmd_pool(&self) -> &CommandPool {
        &self.cmd_pool
    }
    pub fn device(&self) -> &Device {
        self.vulkan.device()
    }
    pub fn destroy(self) -> VulkanContext {
        let Self { vulkan, .. } = self;
        vulkan
    }
    pub fn rerecord_all_cmd_buffers(&mut self)->Result<(),Error>{
        Ok(for image_idx in self.swapchain().iter_images(){
            self.record_cmd_buffer(image_idx)?
        })
    }
    pub fn record_cmd_buffer(&mut self, image_idx:SwapchainImageIdx)->Result<(),Error>{
        let Self{ command_buffers, pipeline, render_pass,descriptors, .. } = self;
        let command_buffer = &mut command_buffers[image_idx.get_usize()];
        command_buffer.reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
        pipeline.record_compute_cmd_buffer(command_buffer)?;
        command_buffer
            .render_pass(render_pass, render_pass.framebuffer(image_idx), render_pass.swapchain().render_area(), &Self::CLEAR_VALUES);
        pipeline.record_cmd_buffer(command_buffer,image_idx,descriptors,render_pass)?;
        command_buffer
            .end_render_pass()
            .end()?;
        Ok(())
    }
    pub fn command_buffer(&self, image_idx:SwapchainImageIdx)->&CommandBuffer{
        &self.command_buffers[image_idx.get_usize()]
    }
    pub fn command_buffer_mut(&mut self, image_idx:SwapchainImageIdx)->&mut CommandBuffer{
        &mut self.command_buffers[image_idx.get_usize()]
    }
    pub fn swapchain(&self) -> &SwapChain {
        self.render_pass.swapchain()
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain().extent()
    }
    pub fn recreate(&mut self) -> Result<(), failure::Error> {
        self.render_pass=self.vulkan.create_single_render_pass()?;
        self.pipeline.recreate(&self.render_pass)?;
        self.descriptors = self.descriptors_builder.build(self.render_pass.swapchain())?;
        let missing_buffers = self.swapchain().len() - self.command_buffers.len();
        if 0 < missing_buffers{
            self.command_buffers.append(&mut self.cmd_pool().create_command_buffers(missing_buffers as u32)?)
        }
        Ok(())
    }

    pub fn render(&mut self, rerecord_cmd:bool, player:&Player) -> Result<bool, failure::Error> {
        let Self{ command_buffers, pipeline, render_pass, vulkan,descriptors, uniforms_binding, .. } = self;
        let fence = vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = render_pass.swapchain().acquire_next_image(None, Some(image_available), None)?;
        let render_finished = vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        let command_buffer = &mut command_buffers[image_idx.get_usize()];
        if rerecord_cmd{
            pipeline.record_cmd_buffer(command_buffer,image_idx,descriptors, render_pass)?
        }
        descriptors.uniform_as_slice_mut(image_idx, *uniforms_binding).copy_from_slice(std::slice::from_ref(player.mvp_uniforms()));
        pipeline.update_uniforms(image_idx, player);

        command_buffer.submit(&[(image_available, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)],
                              std::slice::from_ref(render_finished),
                              Some(fence))?;
        let result = render_pass.swapchain().present(std::slice::from_ref(render_finished), image_idx);
        let is_resized = match result {
            Ok(is_suboptimal) => is_suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(vk::Result::SUBOPTIMAL_KHR) => true,
            err => err?
        };
        vulkan.frames_in_flight_mut().rotate();
        Ok(is_suboptimal || is_resized)
    }
}