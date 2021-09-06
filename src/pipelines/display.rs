
use crate::render::device::{Device, QUEUE_IDX_GRAPHICS, QUEUE_IDX_COMPUTE, QUEUE_IDX_AMBIENT_COMPUTE};




use crate::render::command_pool::{CommandPool, CommandBuffer};



use crate::render::swap_chain::{SwapChain, SwapchainImageIdx};
use crate::render::vulkan_context::VulkanContext;
use failure::Error;
use ash::vk;









use crate::render::single_render_pass::SingleRenderPass;
use crate::render::descriptors::{Descriptors, DescriptorsBuilder, UniformBufferBinding, DescriptorsBuilderLocked};

use crate::pipelines::player::Player;
use crate::pipelines::mvp_uniforms::MvpUniforms;
use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::pipelines::renderable::{RenderResources, Renderable};
use crate::pipelines::computable::{Computable, ComputeResources};
use std::sync::Arc;
use crate::render::fence::Fence;


pub struct Display<P: RenderResources, C:ComputeResources, A:ComputeResources>{
    graphics_command_buffers: Vec<CommandBuffer>,
    compute_command_buffer: CommandBuffer,
    compute_background_command_buffer: CommandBuffer,
    compute_fence:Fence,
    graphics_pipeline:P::Render,
    compute_pipeline:C::Compute,
    compute_background_pipeline:A::Compute,
    render_pass:SingleRenderPass,
    descriptors:Descriptors,
    descriptors_builder:DescriptorsBuilderLocked,
    foundations: Foundations,
    compute_cmd_pool: CommandPool,
    graphics_cmd_pool: CommandPool,
    compute_background_cmd_pool: CommandPool,
    uniforms_binding:UniformBufferBinding<MvpUniforms,1>,
    vulkan: VulkanContext,
}
impl <P: RenderResources, C:ComputeResources, A:ComputeResources> Display<P,C,A> {
    const CLEAR_VALUES: [vk::ClearValue; 2] = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [145./256., 239./256., 2553./256., 1.0],
        },
    }, vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.,
            stencil: 0,
        },
    }];

    pub fn graphics_pipeline(&self) -> &P::Render{
        &self.graphics_pipeline
    }
    pub fn compute_pipeline(&self) -> &C::Compute{
        &self.compute_pipeline
    }
    pub fn compute_background_pipeline(&self) -> &A::Compute{
        &self.compute_background_pipeline
    }
    pub fn graphics_pipeline_mut(&mut self) -> &mut P::Render{
        &mut self.graphics_pipeline
    }
    pub fn new(vulkan: VulkanContext, player:&Player,
               render:impl FnOnce(&CommandPool, &FoundationInitializer)->Result<P,failure::Error>,
               compute:impl FnOnce(&CommandPool, &FoundationInitializer)->Result<C,failure::Error>,
               compute_background:impl FnOnce(&CommandPool, &FoundationInitializer)->Result<A,failure::Error>) -> Result<Self, failure::Error> {
        let render_pass = vulkan.create_single_render_pass()?;
        let graphics_cmd_pool = CommandPool::new(vulkan.device(),QUEUE_IDX_GRAPHICS, true)?;
        let compute_cmd_pool = CommandPool::new(vulkan.device(),QUEUE_IDX_COMPUTE, true)?;
        let compute_background_cmd_pool = CommandPool::new(vulkan.device(),QUEUE_IDX_AMBIENT_COMPUTE, true)?;
        let mut descriptors_builder = DescriptorsBuilder::new();
        let foundations = FoundationInitializer::new(&graphics_cmd_pool)?;
        let uniforms_binding = descriptors_builder.singleton_uniform_buffer(player.mvp_uniforms());
        let _ = descriptors_builder.storage_buffer(foundations.global_mutables().gpu());
        let render_resources = render(&graphics_cmd_pool,&foundations)?;
        let compute_resources = compute(&compute_cmd_pool,&foundations)?;
        let compute_background_resources = compute_background(&compute_background_cmd_pool,&foundations)?;
        render_resources.create_descriptors(&mut descriptors_builder, &foundations)?;
        let descriptors_builder = descriptors_builder.make_layout(graphics_cmd_pool.device())?;
        let foundations = foundations.build()?;
        let graphics_pipeline = render_resources.make_renderable(&graphics_cmd_pool, &render_pass,&descriptors_builder, &foundations)?;
        let compute_pipeline = compute_resources.make_computable(&compute_cmd_pool,&foundations)?;
        let compute_background_pipeline = compute_background_resources.make_computable(&compute_background_cmd_pool,&foundations)?;
        let descriptors = descriptors_builder.build(render_pass.swapchain())?;
        let graphics_command_buffers = graphics_cmd_pool.create_command_buffers(render_pass.framebuffers_len() as u32)?;
        let compute_background_command_buffer = compute_background_cmd_pool.create_command_buffer()?;
        let compute_command_buffer = compute_cmd_pool.create_command_buffer()?;
        let display = Self {
            compute_fence: Fence::new(compute_cmd_pool.device(),true)?,
            descriptors_builder,
            descriptors,
            graphics_command_buffers,
            graphics_pipeline,
            compute_command_buffer,
            compute_pipeline,
            compute_background_command_buffer,
            compute_background_cmd_pool,
            compute_background_pipeline,
            render_pass,
            foundations,
            compute_cmd_pool,
            graphics_cmd_pool,
            vulkan,
            uniforms_binding
        };
        Ok(display)
    }
    pub fn graphics_cmd_pool(&self) -> &CommandPool {
        &self.graphics_cmd_pool
    }
    pub fn compute_cmd_pool(&self) -> &CommandPool {
        &self.compute_cmd_pool
    }
    pub fn device(&self) -> &Device {
        self.vulkan.device()
    }
    pub fn foundations(&self) -> &Foundations {
        &self.foundations
    }
    pub fn foundations_mut(&mut self) -> &mut Foundations {
        &mut self.foundations
    }
    pub fn destroy(self) -> VulkanContext {
        let Self { vulkan, .. } = self;
        vulkan
    }
    pub fn rerecord_all_graphics_cmd_buffers(&mut self)->Result<(),Error>{
        Ok(for image_idx in self.swapchain().iter_images(){
            self.record_graphics_cmd_buffer(image_idx)?
        })
    }
    pub fn record_compute_cmd_buffer(&mut self)->Result<(),Error>{
        let Self{ compute_command_buffer, compute_pipeline, foundations, .. } = self;
        compute_command_buffer.reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
        compute_pipeline.record_compute_cmd_buffer(compute_command_buffer, foundations)?;
        compute_command_buffer.end()?;
        Ok(())
    }
    pub fn record_background_compute_cmd_buffer(&mut self)->Result<(),Error>{
        let Self{ compute_background_command_buffer, compute_background_pipeline, foundations, .. } = self;
        compute_background_command_buffer.reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
        compute_background_pipeline.record_compute_cmd_buffer(compute_background_command_buffer, foundations)?;
        compute_background_command_buffer.end()?;
        Ok(())
    }
    pub fn record_graphics_cmd_buffer(&mut self, image_idx:SwapchainImageIdx)->Result<(),Error>{
        let Self{ graphics_command_buffers: command_buffers, graphics_pipeline, render_pass,descriptors, foundations, .. } = self;
        let command_buffer = &mut command_buffers[image_idx.get_usize()];
        command_buffer.reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
        graphics_pipeline.record_compute_cmd_buffer(command_buffer, foundations)?;
        command_buffer
            .render_pass(render_pass, render_pass.framebuffer(image_idx), render_pass.swapchain().render_area(), &Self::CLEAR_VALUES);
        graphics_pipeline.record_cmd_buffer(command_buffer,image_idx,descriptors,render_pass, foundations)?;
        command_buffer
            .end_render_pass()
            .end()?;
        Ok(())
    }
    pub fn command_buffer(&self, image_idx:SwapchainImageIdx)->&CommandBuffer{
        &self.graphics_command_buffers[image_idx.get_usize()]
    }
    pub fn command_buffer_mut(&mut self, image_idx:SwapchainImageIdx)->&mut CommandBuffer{
        &mut self.graphics_command_buffers[image_idx.get_usize()]
    }
    pub fn swapchain(&self) -> &SwapChain {
        self.render_pass.swapchain()
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain().extent()
    }
    pub fn recreate_graphics(&mut self) -> Result<(), failure::Error> {
        self.render_pass=self.vulkan.create_single_render_pass()?;
        self.graphics_pipeline.recreate(&self.render_pass,self.foundations.specialization_constants())?;
        self.descriptors = self.descriptors_builder.build(self.render_pass.swapchain())?;
        let missing_buffers = self.swapchain().len() - self.graphics_command_buffers.len();
        if 0 < missing_buffers{
            self.graphics_command_buffers.append(&mut self.graphics_cmd_pool().create_command_buffers(missing_buffers as u32)?)
        }
        Ok(())
    }

    pub fn compute(&mut self, player:&mut Player) -> Result<(), failure::Error> {
        let Self{ compute_command_buffer, foundations,compute_pipeline
            , compute_background_command_buffer,compute_background_pipeline ,compute_fence
            , vulkan, .. } = self;
        if compute_fence.is_signaled()? {
            compute_fence.reset()?;
            compute_pipeline.update_uniforms( player, foundations);
            compute_command_buffer.submit(&[], &[], Some(compute_fence))?;
            compute_background_pipeline.update_uniforms(player, foundations);
            compute_background_command_buffer.submit(&[], &[], None)?;
        }
        Ok(())
    }
    pub fn render(&mut self, _rerecord_cmd:bool, player:&mut Player) -> Result<bool, failure::Error> {
        let Self{ graphics_command_buffers: command_buffers, graphics_pipeline, render_pass,foundations: _, vulkan,descriptors, uniforms_binding, .. } = self;
        let fence = vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = render_pass.swapchain().acquire_next_image(None, Some(image_available), None)?;
        let render_finished = vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        let command_buffer = &mut command_buffers[image_idx.get_usize()];
        // if rerecord_cmd{
        //     pipeline.record_cmd_buffer(command_buffer,image_idx,descriptors, render_pass, foundations)?
        // }
        descriptors.uniform_as_slice_mut(image_idx, *uniforms_binding).copy_from_slice(std::slice::from_ref(player.mvp_uniforms()));
        graphics_pipeline.update_uniforms(image_idx, player);

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