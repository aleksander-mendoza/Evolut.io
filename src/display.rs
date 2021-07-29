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
use crate::render::buffer::{Buffer};
use crate::render::fence::Fence;
use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::texture::{StageTexture, Dim2D, TextureView};
use crate::render::sampler::Sampler;
use crate::render::framebuffer::Framebuffer;
use crate::blocks::Face;
use crate::render::single_render_pass::SingleRenderPass;
use crate::render::descriptors::Descriptors;
use std::marker::PhantomData;


pub trait Renderable<U:Copy>:Sized{
    fn new(cmd_pool:&CommandPool,render_pass:&SingleRenderPass,uniform_data:&U)->Result<Self,failure::Error>;
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx:SwapchainImageIdx, render_pass:&SingleRenderPass)->Result<(),Error>;
    fn update_uniforms(&mut self, image_idx:SwapchainImageIdx, uniform_data:&U);
    fn recreate(&mut self, render_pass: &SingleRenderPass) -> Result<(), Error>;
}

pub struct Display<U:Copy,P:Renderable<U>>{
    command_buffers: Vec<CommandBuffer>,
    pipeline:P,
    render_pass:SingleRenderPass,
    cmd_pool: CommandPool,
    vulkan: VulkanContext,
    _p:PhantomData<U>
}
impl <U:Copy,P:Renderable<U>> Display<U,P> {
    pub fn new(vulkan: VulkanContext, uniform_data:&U) -> Result<Self, failure::Error> {
        let render_pass = vulkan.create_single_render_pass()?;
        let cmd_pool = CommandPool::new(vulkan.device(),true)?;
        let pipeline = P::new(&cmd_pool, &render_pass, uniform_data)?;
        let command_buffers = cmd_pool.create_command_buffers(render_pass.framebuffers_len() as u32)?;
        Ok(Self {
            command_buffers,
            pipeline,
            render_pass,
            cmd_pool,
            vulkan,
            _p:PhantomData
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
        let Self{ command_buffers, pipeline, render_pass, .. } = self;
        let command_buffer = &mut command_buffers[image_idx.get_usize()];
        pipeline.record_cmd_buffer(command_buffer,image_idx,render_pass)
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
        let missing_buffers = self.swapchain().len() - self.command_buffers.len();
        if 0 < missing_buffers{
            self.command_buffers.append(&mut self.cmd_pool().create_command_buffers(missing_buffers as u32)?)
        }
        Ok(())
    }

    pub fn render(&mut self, rerecord_cmd:bool, uniform_data:&U) -> Result<bool, failure::Error> {
        let Self{ command_buffers, pipeline, render_pass, cmd_pool, vulkan, _p } = self;
        let fence = vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = render_pass.swapchain().acquire_next_image(None, Some(image_available), None)?;
        let render_finished = vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        let command_buffer = &mut command_buffers[image_idx.get_usize()];
        if rerecord_cmd{
            pipeline.record_cmd_buffer(command_buffer,image_idx,render_pass)?
        }
        pipeline.update_uniforms(image_idx, uniform_data);

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